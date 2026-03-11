"""
pdf_ingest.py — Multimodal PDF ingestion using unstructured (fast strategy) for figure detection.
No deep learning models (YOLOX) are used. Text and table extraction unchanged.
"""

import io
import base64
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz                        # PyMuPDF ≥ 1.23
import pdfplumber
import numpy as np
from PIL import Image
from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

from rag.cohere_store import cohere_vector_store
from ingest.image_store import upload_page_image

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Text ──────────────────────────────────────────────────────────────────────
TEXT_CHUNK_SIZE    = 800
TEXT_CHUNK_OVERLAP = 200

# ── Rendering ─────────────────────────────────────────────────────────────────
RENDER_DPI = 200          # DPI for all pixmap renders

# ── unstructured settings (NO YOLOX) ──────────────────────────────────────────
UNSTRUCTURED_STRATEGY = os.getenv("UNSTRUCTURED_STRATEGY", "fast").lower()
# "fast" uses rule‑based layout analysis – no deep learning model.
# If you want a different hi‑res model later, set this to "hi_res" and
# set UNSTRUCTURED_MODEL to e.g. "detectron2_onnx".

FIGURE_CATEGORIES = {"Image", "Figure", "Picture", "Illustration", "Graph"}
CAPTION_CATEGORIES = {"FigureCaption", "Caption"}

# ── Figure bbox merging (fixes split sub-figures) ─────────────────────────────
MERGE_GAP_PX = 60         # at 200 DPI, 60px ≈ 8.5mm

# ── Figure quality filters (same as before) ───────────────────────────────────
MIN_FIGURE_AREA   = 80 * 80   # px² after crop + trim
MIN_CONTENT_RATIO = 0.02      # fraction non-white pixels
MAX_FIGURE_ASPECT = 8.0

# ── Duplicate suppression ─────────────────────────────────────────────────────
DEDUP_HASH_DISTANCE = 8

# ── Caption matching ──────────────────────────────────────────────────────────
CAPTION_MAX_DIST_PT = 80      # pt — max vertical gap in PDF space

# ── Vision ────────────────────────────────────────────────────────────────────
VISION_MODEL      = "gpt-4o-mini"
VISION_MAX_TOKENS = 600
MAX_IMAGE_SIDE    = 768
VISION_DETAIL     = "high"
VISION_WORKERS    = 8

# ── Tables ────────────────────────────────────────────────────────────────────
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

# ── Local image saving ────────────────────────────────────────────────────────
SAVE_IMAGES_LOCAL = os.getenv("SAVE_IMAGES_LOCAL", "false").lower() == "true"
LOCAL_IMAGE_DIR   = Path(os.getenv("LOCAL_IMAGE_DIR", "extracted_images"))

# PDF passwords to try for encrypted files
PDF_PASSWORDS = ["", "password", "1234", "admin"]

# Caption text pattern (English fallback; spatial matching is primary)
_CAPTION_RE = re.compile(r"fig(?:ure)?\.?\s*\d+", re.I)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FigureTask:
    """One figure ready for vision processing."""
    img:        Image.Image
    page_num:   int
    caption:    str
    label:      str
    fig_idx:    int
    layer:      str           # "unstructured-fast"


# ═══════════════════════════════════════════════════════════════════════════════
# PDF OPEN — encryption handling (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _open_pdf(file_bytes: bytes) -> Optional[fitz.Document]:
    """Open PDF, attempt decryption with common passwords if encrypted."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"Cannot open PDF: {e}")
        return None

    if not doc.is_encrypted:
        return doc

    for pwd in PDF_PASSWORDS:
        if doc.authenticate(pwd):
            logger.info(f"PDF decrypted with password='{pwd}'")
            return doc

    logger.error("PDF encrypted — no known password worked")
    doc.close()
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _pil_from_bytes(raw: bytes, ext: str = "") -> Optional[Image.Image]:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img.convert("RGB")
    except Exception as e:
        logger.debug(f"PIL decode failed (ext={ext}): {e}")
        return None


def _pix_to_pil(pix: fitz.Pixmap) -> Image.Image:
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _render_rect(page: fitz.Page, rect: fitz.Rect) -> fitz.Pixmap:
    scale = RENDER_DPI / 72
    mat   = fitz.Matrix(scale, scale)
    return page.get_pixmap(matrix=mat, clip=rect, alpha=False)


def _render_full_page(page: fitz.Page) -> fitz.Pixmap:
    scale = RENDER_DPI / 72
    mat   = fitz.Matrix(scale, scale)
    return page.get_pixmap(matrix=mat, alpha=False)


def _content_ratio(img: Image.Image) -> float:
    arr = np.array(img.convert("L"))
    return float((arr < 245).sum()) / max(arr.size, 1)


def _trim_whitespace(img: Image.Image) -> Image.Image:
    arr  = np.array(img.convert("L"))
    mask = arr < 245
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return img
    return img.crop((int(cols[0]), int(rows[0]),
                     int(cols[-1]) + 1, int(rows[-1]) + 1))


def _is_valid_figure(img: Image.Image) -> bool:
    w, h = img.size
    if w * h < MIN_FIGURE_AREA:
        return False
    if _content_ratio(img) < MIN_CONTENT_RATIO:
        return False
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_FIGURE_ASPECT:
        return False
    return True


def _resize_for_vision(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_SIDE:
        return img
    scale = MAX_IMAGE_SIDE / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Perceptual hash ───────────────────────────────────────────────────────────

def _phash(img: Image.Image) -> int:
    small = np.array(img.convert("L").resize((8, 8), Image.LANCZOS), dtype=float)
    bits  = (small >= small.mean()).flatten()
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def _is_duplicate(img: Image.Image, seen: List[int]) -> bool:
    h = _phash(img)
    for prev in seen:
        if bin(h ^ prev).count("1") <= DEDUP_HASH_DISTANCE:
            return True
    seen.append(h)
    return False


# ── Local saving ──────────────────────────────────────────────────────────────

def _save_image_locally(img: Image.Image, thread_id: str, page_num: int, fig_idx: int) -> Optional[Path]:
    if not SAVE_IMAGES_LOCAL:
        return None
    thread_dir = LOCAL_IMAGE_DIR / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    filename = f"page_{page_num:03d}_fig_{fig_idx}_{unique_id}.jpg"
    filepath = thread_dir / filename
    try:
        img.convert("RGB").save(filepath, "JPEG", quality=88)
        logger.debug(f"Saved locally: {filepath}")
        return filepath
    except Exception as e:
        logger.warning(f"Failed to save image locally: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# UNSTRUCTURED‑SPECIFIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _unstructured_bbox_to_pixels(
    el,
    page_width_pts:  float,
    page_height_pts: float,
    render_w_px:     int,
    render_h_px:     int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert an unstructured element's coordinate metadata to pixel coords
    in the rendered pixmap. Supports PointSpace, PixelSpace, and normalized.
    """
    coords = getattr(el.metadata, "coordinates", None)
    if coords is None or not coords.points:
        return None

    pts    = coords.points
    xs     = [p[0] for p in pts]
    ys     = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)

    # Detect coordinate system
    system     = getattr(coords, "system", None)
    sys_name   = system.__class__.__name__ if system else "PointSpace"

    # Detect normalised (0-1) coords
    if x1 <= 1.0 and y1 <= 1.0:
        x0 *= render_w_px
        x1 *= render_w_px
        y0 *= render_h_px
        y1 *= render_h_px
        return (int(x0), int(y0), int(x1), int(y1))

    if sys_name == "PointSpace" or (x1 > 1 and x1 <= page_width_pts * 1.05):
        # Scale from PDF points to render pixels
        sx = render_w_px / page_width_pts
        sy = render_h_px / page_height_pts
        return (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
    else:
        # PixelSpace – assume unstructured extracted at 200 DPI (default)
        scale = RENDER_DPI / 200.0
        return (
            int(x0 * scale), int(y0 * scale),
            int(x1 * scale), int(y1 * scale),
        )


def _boxes_close(
    a: Tuple[int,int,int,int],
    b: Tuple[int,int,int,int],
    gap: int,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + gap < bx0 or bx1 + gap < ax0 or
        ay1 + gap < by0 or by1 + gap < ay0
    )


def _union_box(
    a: Tuple[int,int,int,int],
    b: Tuple[int,int,int,int],
) -> Tuple[int,int,int,int]:
    return (min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3]))


def _merge_figure_bboxes(
    bboxes: List[Tuple[int,int,int,int]],
) -> List[Tuple[int,int,int,int]]:
    """Merge any two bboxes within MERGE_GAP_PX of each other."""
    if not bboxes:
        return []
    clusters = list(bboxes)
    changed  = True
    while changed:
        changed = False
        merged: List[Tuple[int,int,int,int]] = []
        used = [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            cur = clusters[i]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if _boxes_close(cur, clusters[j], MERGE_GAP_PX):
                    cur = _union_box(cur, clusters[j])
                    used[j] = True
                    changed = True
            merged.append(cur)
        clusters = merged
    return clusters


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE EXTRACTION USING UNSTRUCTURED (FAST STRATEGY – NO YOLOX)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_image_docs(file_bytes: bytes, thread_id: str) -> List[Document]:
    """
    Use unstructured (fast strategy) to detect figure regions,
    then render, describe, and upload. No deep learning models are used.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as e:
        raise RuntimeError(
            "unstructured not installed.\n"
            "Run: pip install 'unstructured[pdf]'"
        ) from e

    tmp_path = None
    elements = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        logger.info(f"Running unstructured partition_pdf (strategy={UNSTRUCTURED_STRATEGY})…")
        # No hi_res_model_name is passed – the strategy determines the backend.
        # "fast" uses rule‑based layout analysis, no model required.
        elements = partition_pdf(
            filename=tmp_path,
            strategy=UNSTRUCTURED_STRATEGY,
            infer_table_structure=False,   # we handle tables separately
            extract_images_in_pdf=False,
        )
        logger.info(f"unstructured returned {len(elements)} elements")
    except Exception as e:
        logger.error(f"partition_pdf failed: {e}")
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not elements:
        return []

    # Group elements by page
    figures_by_page: Dict[int, List] = {}
    captions_by_page: Dict[int, List] = {}

    for el in elements:
        page = getattr(el.metadata, "page_number", None)
        if page is None:
            continue
        if el.category in FIGURE_CATEGORIES:
            figures_by_page.setdefault(page, []).append(el)
        elif el.category in CAPTION_CATEGORIES:
            captions_by_page.setdefault(page, []).append(el)
        elif el.category in {"NarrativeText", "Text"}:
            text = (el.text or "").strip()
            if _CAPTION_RE.match(text):
                captions_by_page.setdefault(page, []).append(el)

    if not figures_by_page:
        logger.info("No figure elements found by unstructured.")
        return []

    # Open PDF with PyMuPDF for rendering
    doc = _open_pdf(file_bytes)
    if doc is None:
        return []

    seen_hashes: List[int] = []
    all_tasks: List[FigureTask] = []

    scale = RENDER_DPI / 72

    # Process each page with figures
    for page_num, fig_elements in sorted(figures_by_page.items()):
        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= doc.page_count:
            continue
        fitz_page = doc[page_idx]
        page_w_pts = fitz_page.rect.width
        page_h_pts = fitz_page.rect.height
        render_w   = int(page_w_pts * scale)
        render_h   = int(page_h_pts * scale)

        # Convert figure bboxes to pixels
        raw_bboxes = []
        for el in fig_elements:
            bbox = _unstructured_bbox_to_pixels(
                el, page_w_pts, page_h_pts, render_w, render_h
            )
            if bbox:
                raw_bboxes.append(bbox)

        if not raw_bboxes:
            continue

        # Merge nearby bboxes
        merged_bboxes = _merge_figure_bboxes(raw_bboxes)

        # Convert captions to candidates for spatial matching
        caption_candidates = []
        for cap_el in captions_by_page.get(page_num, []):
            cap_bbox = _unstructured_bbox_to_pixels(
                cap_el, page_w_pts, page_h_pts, render_w, render_h
            )
            if cap_bbox:
                _, cy0, _, cy1 = cap_bbox
                caption_candidates.append({
                    "text": (cap_el.text or "").strip(),
                    "center_y": (cy0 + cy1) / 2,
                })

        # Render the page once (all crops from same pixmap)
        pix = fitz_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        for fig_idx, bbox in enumerate(merged_bboxes, start=1):
            x0, y0, x1, y1 = bbox

            # Clamp to page bounds
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(render_w, x1); y1 = min(render_h, y1)
            if x1 <= x0 or y1 <= y0:
                continue

            # Crop and trim whitespace
            cropped = page_img.crop((x0, y0, x1, y1))
            cropped = _trim_whitespace(cropped)

            if not _is_valid_figure(cropped):
                logger.debug(f"Page {page_num} fig {fig_idx}: invalid figure, skipping")
                continue
            if _is_duplicate(cropped, seen_hashes):
                logger.debug(f"Page {page_num} fig {fig_idx}: duplicate, skipping")
                continue

            # Match caption
            fig_center_y = (y0 + y1) / 2
            caption = ""
            if caption_candidates:
                best, best_dist = "", float("inf")
                for cand in caption_candidates:
                    dist = abs(fig_center_y - cand["center_y"])
                    if dist < best_dist:
                        best_dist = dist
                        best = cand["text"]
                # Convert pt threshold to pixels
                if best_dist <= CAPTION_MAX_DIST_PT * (RENDER_DPI / 72):
                    caption = best

            label = f"Page {page_num} — Figure {fig_idx} ({cropped.width}×{cropped.height}px)"

            all_tasks.append(FigureTask(
                img=cropped,
                page_num=page_num,
                caption=caption,
                label=label,
                fig_idx=fig_idx,
                layer="unstructured-fast",
            ))

    doc.close()
    logger.info(f"Images: {len(all_tasks)} figures queued for vision")

    if not all_tasks:
        return []

    # Parallel vision + upload + local save
    def _run_task(task: FigureTask) -> Optional[Document]:
        description = _describe_figure(task)
        if not description:
            return None

        content_parts = [f"[{task.label}]"]
        if task.caption:
            content_parts.append(f"Caption: {task.caption}")
        content_parts.append(description)

        try:
            image_url = upload_page_image(
                task.img, thread_id, task.page_num, task.fig_idx
            )
        except Exception as e:
            logger.warning(f"S3 upload failed — {task.label}: {e}")
            image_url = None

        if SAVE_IMAGES_LOCAL:
            _save_image_locally(task.img, thread_id, task.page_num, task.fig_idx)

        if image_url is None:
            # still return document without image_ref
            return Document(
                page_content="\n".join(content_parts),
                metadata={
                    "thread_id":    thread_id,
                    "content_type": "image",
                    "page":         task.page_num,
                    "figure_index": task.fig_idx,
                    "source":       "pdf",
                    "caption":      task.caption,
                    "layer":        task.layer,
                },
            )

        return Document(
            page_content="\n".join(content_parts),
            metadata={
                "thread_id":    thread_id,
                "content_type": "image",
                "page":         task.page_num,
                "figure_index": task.fig_idx,
                "source":       "pdf",
                "caption":      task.caption,
                "layer":        task.layer,
                "image_ref":    image_url,
            },
        )

    image_docs: List[Document] = []
    with ThreadPoolExecutor(max_workers=VISION_WORKERS) as executor:
        future_map = {executor.submit(_run_task, t): t for t in all_tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
            except Exception as e:
                logger.warning(f"Task error — {task.label}: {e}")
                continue
            if result:
                image_docs.append(result)
                logger.info(f"  ✅ {task.label} [{task.layer}]")

    logger.info(f"Images: {len(image_docs)} documents created")
    return image_docs


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION (unchanged — PyMuPDF)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text_chunks(file_bytes: bytes, thread_id: str) -> List[Document]:
    doc = _open_pdf(file_bytes)
    if doc is None:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
    )
    all_docs: List[Document] = []

    for page_num, page in enumerate(doc, start=1):
        try:
            blocks = sorted(
                page.get_text("blocks"),
                key=lambda b: (round(b[1] / 10) * 10, b[0]),
            )
            page_text = "\n".join(
                b[4].strip() for b in blocks
                if b[6] == 0 and b[4].strip()
            )
            if len(page_text) < 50:
                continue
            for chunk in splitter.split_text(page_text):
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "thread_id":    thread_id,
                        "content_type": "text",
                        "page":         page_num,
                        "source":       "pdf",
                    },
                ))
        except Exception as e:
            logger.warning(f"Page {page_num}: text extraction failed — {e}")

    doc.close()
    logger.info(f"Text: {len(all_docs)} chunks")
    return all_docs


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE EXTRACTION (unchanged — pdfplumber)
# ═══════════════════════════════════════════════════════════════════════════════

def _list_to_markdown(table: List[List]) -> str:
    if not table or len(table) < MIN_TABLE_ROWS:
        return ""
    cleaned = [[str(c).strip() if c else "" for c in row] for row in table]
    if len(cleaned[0]) < MIN_TABLE_COLS:
        return ""
    hdr  = cleaned[0]
    sep  = ["---"] * len(hdr)
    lines = [
        "| " + " | ".join(hdr) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in cleaned[1:]:
        p = row + [""] * (len(hdr) - len(row))
        lines.append("| " + " | ".join(p[:len(hdr)]) + " |")
    return "\n".join(lines)


def _extract_table_docs(file_bytes: bytes, thread_id: str) -> List[Document]:
    docs: List[Document] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    for idx, table in enumerate(page.extract_tables() or [], start=1):
                        md = _list_to_markdown(table)
                        if not md:
                            continue
                        docs.append(Document(
                            page_content=f"[Page {page_num} — Table {idx}]\n{md}",
                            metadata={
                                "thread_id":    thread_id,
                                "content_type": "table",
                                "page":         page_num,
                                "table_index":  idx,
                                "source":       "pdf",
                            },
                        ))
                except Exception as e:
                    logger.warning(f"Page {page_num}: table extraction failed — {e}")
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    logger.info(f"Tables: {len(docs)} tables extracted")
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# VISION DESCRIPTION (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

_EQ_HINT = (
    "This figure likely contains mathematical equations or notation. "
    "Transcribe visible formulas using LaTeX notation where possible and "
    "explain what each formula represents conceptually.\n"
)

def _describe_figure(task: FigureTask) -> Optional[str]:
    try:
        img = task.img
        b64 = _to_base64_jpeg(_resize_for_vision(img))

        arr      = np.array(img.convert("L"))
        eq_hint  = _EQ_HINT if (len(np.unique(arr)) / 256.0) < 0.15 else ""
        cap_hint = f'\nFigure caption: "{task.caption}"\n' if task.caption else ""

        prompt = (
            f"You are analyzing a figure from page {task.page_num} of a PDF."
            f"{cap_hint}{eq_hint}\n"
            "Respond in EXACTLY this structure:\n\n"
            "FIGURE TYPE: <bar chart | line chart | scatter plot | heatmap | "
            "flowchart | architecture diagram | attention map | network diagram | "
            "photograph | screenshot | infographic | table | equation | other>\n"
            "KEY FINDING: <one sentence — the single most important insight>\n"
            "DESCRIPTION: <thorough description — include all visible labels, "
            "axis names and ranges, legend entries, color meanings, annotated "
            "regions, component names, data trends, and what every visual "
            "element represents. Detailed enough that someone without the "
            "image can fully answer questions about it.>\n"
            "SEARCHABLE TERMS: <comma-separated domain terms, metric names, "
            "model names, dataset names, acronyms visible in the figure>"
        )

        response = _openai.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=VISION_MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/jpeg;base64,{b64}",
                            "detail": VISION_DETAIL,
                        },
                    },
                ],
            }],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning(f"Vision API failed — {task.label}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_pdf_multimodal(
    file_bytes: bytes,
    thread_id:  str,
    *,
    extract_text:   bool = True,
    extract_images: bool = True,
    extract_tables: bool = True,
) -> dict:
    all_docs: List[Document] = []
    counts = {"text": 0, "images": 0, "tables": 0}

    if extract_text:
        docs = _extract_text_chunks(file_bytes, thread_id)
        all_docs.extend(docs)
        counts["text"] = len(docs)

    if extract_tables:
        docs = _extract_table_docs(file_bytes, thread_id)
        all_docs.extend(docs)
        counts["tables"] = len(docs)

    if extract_images:
        docs = _extract_image_docs(file_bytes, thread_id)
        all_docs.extend(docs)
        counts["images"] = len(docs)

    if not all_docs:
        logger.warning(f"No content extracted from PDF for thread={thread_id}")
        return {**counts, "total": 0}

    cohere_vector_store.add_texts(
        texts=[d.page_content for d in all_docs],
        metadatas=[d.metadata for d in all_docs],
    )

    counts["total"] = len(all_docs)
    logger.info(
        f"✅ Multimodal ingestion complete — thread={thread_id} | "
        f"text={counts['text']} table={counts['tables']} "
        f"image={counts['images']} total={counts['total']} | "
        f"store=crag_multimodal (Cohere)"
    )
    return counts


def ingest_pdf(file_bytes: bytes, thread_id: str) -> None:
    ingest_pdf_multimodal(file_bytes, thread_id)