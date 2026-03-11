# """
# pdf_ingest.py — Universal multimodal PDF ingestion using PyMuPDF only.

# ═══════════════════════════════════════════════════════════════════════════════
# WHY PyMuPDF-ONLY
# ═══════════════════════════════════════════════════════════════════════════════
# PyMuPDF (fitz) gives us direct access to every PDF internal structure:
#   • Embedded raster images     → doc.extract_image(xref)
#   • Vector drawings            → page.get_drawings()
#   • Text with bbox + font info → page.get_text("dict")
#   • Page structure / blocks    → page.get_text("blocks")
#   • Form XObjects              → page.get_xobjects()
#   • Annotations                → page.annots()
#   • Encryption                 → doc.authenticate()
#   • OCR (built-in Tesseract)   → page.get_textpage_ocr()

# All coordinate math stays in one consistent PDF-point space.

# FIGURE EXTRACTION STRATEGY — 5 LAYERS
# ═══════════════════════════════════════════════════════════════════════════════

# Every page is analysed through 5 layers in order:

# LAYER A — Embedded raster xobjects
#   doc.extract_image(xref) returns the exact bytes the PDF author embedded.
#   Pixel-perfect, zero rendering cost. Handles PNG, JPEG, JPEG2000, JBIG2.
#   Filters: minimum size, smask skip, cross-page dedup via seen_xref set.
#   Covers: matplotlib figures, photos, screenshots saved into PDF.

# LAYER B — Vector figure clusters
#   page.get_drawings() returns every stroke/fill with its PDF-point bbox.
#   Nearby rects are union-merged (CLUSTER_GAP) into figure clusters.
#   Each cluster is rendered via page.get_pixmap(clip=cluster_rect).
#   Only runs when Layer A found nothing (page has no embedded rasters).
#   Covers: TikZ diagrams, draw.io, architecture diagrams, flowcharts.

# LAYER C — Mixed pages (raster + vector overlay)
#   When a page has both raster xobjects AND significant vector drawings
#   on top of them (annotation arrows, axis labels drawn as paths), we
#   render the page region instead of using raw extracted bytes.
#   The rendered crop composites raster + vector together correctly.
#   Covers: Matplotlib charts with vector annotations, labelled photos.

# LAYER D — Scanned pages
#   When a page has < SCAN_TEXT_THRESHOLD selectable chars AND no raster
#   xobjects were found, the whole page is likely a scanned image.
#   We render the full page and send it to vision.
#   If PyMuPDF is built with Tesseract support, OCR text is also extracted.
#   Covers: scanned books, legacy reports, photographed whiteboards.

# LAYER E — Form XObjects (reusable graphic objects)
#   page.get_xobjects() returns /XObject /Form entries — reusable graphics
#   placed on the page by InDesign, Illustrator, PowerPoint exports.
#   Each is located and rendered from its placement rect.
#   Covers: InDesign exports, Illustrator figures, infographics.

# ADDITIONAL FEATURES
#   • Password-protected PDFs    — common passwords tried automatically
#   • Large PDFs (100+ pages)    — page-by-page streaming, no full-doc RAM load
#   • Corrupt pages              — per-page try/except, bad page ≠ abort
#   • Duplicate figures          — perceptual hash dedup across all pages
#   • Caption matching           — spatial proximity from page text blocks
#   • Table extraction           — pdfplumber (most reliable for tables)
#   • Equation detection         — specialised vision prompt for math-heavy crops
#   • Sub-figure merging         — nearby figure clusters merged before crop
#   • Whitespace trimming        — removes margin bleed from every crop

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════
#   pip install pymupdf pdfplumber pillow openai numpy
#               langchain-core langchain-text-splitters python-dotenv
# """

# import io
# import base64
# import logging
# import os
# import re
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Set, Tuple
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import fitz                        # PyMuPDF ≥ 1.23
# import pdfplumber
# import numpy as np
# from PIL import Image
# from openai import OpenAI
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv

# load_dotenv()

# from rag.cohere_store import cohere_vector_store
# from ingest.image_store import upload_page_image

# logger = logging.getLogger(__name__)

# # ═══════════════════════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════════════════════

# _openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # ── Text ──────────────────────────────────────────────────────────────────────
# TEXT_CHUNK_SIZE    = 800
# TEXT_CHUNK_OVERLAP = 200

# # ── Rendering ─────────────────────────────────────────────────────────────────
# RENDER_DPI = 200          # DPI for all pixmap renders

# # ── Layer A: raster xobject filters ──────────────────────────────────────────
# MIN_XOBJ_WIDTH  = 80      # px — skip tiny icons / bullets
# MIN_XOBJ_HEIGHT = 80      # px

# # ── Layer B: vector cluster tuning ────────────────────────────────────────────
# CLUSTER_GAP      = 40     # pt — rects within this distance are merged
# MIN_CLUSTER_AREA = 2500   # pt² (50×50) — minimum qualifying cluster
# CLUSTER_MAX_ASPECT = 10.0 # skip long thin rules (table borders, dividers)
# CROP_PADDING     = 8      # pt — padding around cluster rect before render

# # ── Layer C: mixed page detection ─────────────────────────────────────────────
# # If vector drawings cover > this fraction of an xobject's area, render
# # the page region (which composites raster + vector) instead of raw bytes
# VECTOR_OVERLAY_THRESHOLD = 0.12

# # ── Layer D: scanned page detection ───────────────────────────────────────────
# SCAN_TEXT_THRESHOLD = 50  # chars — below this = likely scanned page

# # ── Figure quality filters (all layers) ──────────────────────────────────────
# MIN_FIGURE_AREA   = 80 * 80   # px² after crop + trim
# MIN_CONTENT_RATIO = 0.02      # fraction non-white pixels
# MAX_FIGURE_ASPECT = 8.0       # skip extreme horizontal/vertical strips

# # ── Duplicate suppression ─────────────────────────────────────────────────────
# DEDUP_HASH_DISTANCE = 8       # max perceptual hash Hamming distance

# # ── Caption matching ──────────────────────────────────────────────────────────
# CAPTION_MAX_DIST_PT = 80      # pt — max vertical gap figure↔caption in PDF space

# # ── Vision ────────────────────────────────────────────────────────────────────
# VISION_MODEL      = "gpt-4o-mini"
# VISION_MAX_TOKENS = 600
# MAX_IMAGE_SIDE    = 768
# VISION_DETAIL     = "high"
# VISION_WORKERS    = 8

# # ── Tables ────────────────────────────────────────────────────────────────────
# MIN_TABLE_ROWS = 2
# MIN_TABLE_COLS = 2

# # PDF passwords to try for encrypted files
# PDF_PASSWORDS = ["", "password", "1234", "admin"]

# # Caption text pattern (English fallback; spatial matching is primary)
# _CAPTION_RE = re.compile(r"fig(?:ure)?\.?\s*\d+", re.I)


# # ═══════════════════════════════════════════════════════════════════════════════
# # DATA STRUCTURES
# # ═══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class FigureTask:
#     """One figure ready for vision processing."""
#     img:        Image.Image
#     page_num:   int
#     caption:    str
#     label:      str
#     fig_idx:    int
#     layer:      str           # "raster"|"vector"|"mixed"|"scanned"|"form"


# # ═══════════════════════════════════════════════════════════════════════════════
# # PDF OPEN — encryption handling
# # ═══════════════════════════════════════════════════════════════════════════════

# def _open_pdf(file_bytes: bytes) -> Optional[fitz.Document]:
#     """Open PDF, attempt decryption with common passwords if encrypted."""
#     try:
#         doc = fitz.open(stream=file_bytes, filetype="pdf")
#     except Exception as e:
#         logger.error(f"Cannot open PDF: {e}")
#         return None

#     if not doc.is_encrypted:
#         return doc

#     for pwd in PDF_PASSWORDS:
#         if doc.authenticate(pwd):
#             logger.info(f"PDF decrypted with password='{pwd}'")
#             return doc

#     logger.error("PDF encrypted — no known password worked")
#     doc.close()
#     return None


# # ═══════════════════════════════════════════════════════════════════════════════
# # IMAGE UTILITIES
# # ═══════════════════════════════════════════════════════════════════════════════

# def _pil_from_bytes(raw: bytes, ext: str = "") -> Optional[Image.Image]:
#     """Decode raw image bytes → PIL RGB. Returns None on failure."""
#     try:
#         img = Image.open(io.BytesIO(raw))
#         img.load()
#         return img.convert("RGB")
#     except Exception as e:
#         logger.debug(f"PIL decode failed (ext={ext}): {e}")
#         return None


# def _pix_to_pil(pix: fitz.Pixmap) -> Image.Image:
#     """Convert fitz Pixmap → PIL RGB Image."""
#     return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# def _render_rect(
#     page: fitz.Page,
#     rect: fitz.Rect,
# ) -> fitz.Pixmap:
#     """Render a clipped rect of the page at RENDER_DPI."""
#     scale = RENDER_DPI / 72
#     mat   = fitz.Matrix(scale, scale)
#     return page.get_pixmap(matrix=mat, clip=rect, alpha=False)


# def _render_full_page(page: fitz.Page) -> fitz.Pixmap:
#     """Render entire page at RENDER_DPI."""
#     scale = RENDER_DPI / 72
#     mat   = fitz.Matrix(scale, scale)
#     return page.get_pixmap(matrix=mat, alpha=False)


# def _content_ratio(img: Image.Image) -> float:
#     """Fraction of non-white pixels."""
#     arr = np.array(img.convert("L"))
#     return float((arr < 245).sum()) / max(arr.size, 1)


# def _trim_whitespace(img: Image.Image) -> Image.Image:
#     """Remove near-white border rows/cols."""
#     arr  = np.array(img.convert("L"))
#     mask = arr < 245
#     rows = np.where(mask.any(axis=1))[0]
#     cols = np.where(mask.any(axis=0))[0]
#     if not len(rows) or not len(cols):
#         return img
#     return img.crop((int(cols[0]), int(rows[0]),
#                      int(cols[-1]) + 1, int(rows[-1]) + 1))


# def _is_valid_figure(img: Image.Image) -> bool:
#     """Passes minimum area, content ratio, and aspect ratio filters."""
#     w, h = img.size
#     if w * h < MIN_FIGURE_AREA:
#         return False
#     if _content_ratio(img) < MIN_CONTENT_RATIO:
#         return False
#     aspect = max(w, h) / max(min(w, h), 1)
#     if aspect > MAX_FIGURE_ASPECT:
#         return False
#     return True


# def _resize_for_vision(img: Image.Image) -> Image.Image:
#     w, h = img.size
#     if max(w, h) <= MAX_IMAGE_SIDE:
#         return img
#     scale = MAX_IMAGE_SIDE / max(w, h)
#     return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# def _to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
#     buf = io.BytesIO()
#     img.convert("RGB").save(buf, format="JPEG", quality=quality)
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


# # ── Perceptual hash (pure numpy — no external lib) ───────────────────────────

# def _phash(img: Image.Image) -> int:
#     """8×8 average perceptual hash as integer."""
#     small = np.array(img.convert("L").resize((8, 8), Image.LANCZOS), dtype=float)
#     bits  = (small >= small.mean()).flatten()
#     return int(sum(int(b) << i for i, b in enumerate(bits)))


# def _is_duplicate(img: Image.Image, seen: List[int]) -> bool:
#     """True if img is visually similar to any previously seen figure."""
#     h = _phash(img)
#     for prev in seen:
#         if bin(h ^ prev).count("1") <= DEDUP_HASH_DISTANCE:
#             return True
#     seen.append(h)
#     return False


# # ═══════════════════════════════════════════════════════════════════════════════
# # VECTOR CLUSTER EXTRACTION (Layer B core)
# # ═══════════════════════════════════════════════════════════════════════════════

# def _union_rect(a: fitz.Rect, b: fitz.Rect) -> fitz.Rect:
#     return fitz.Rect(min(a.x0,b.x0), min(a.y0,b.y0),
#                      max(a.x1,b.x1), max(a.y1,b.y1))


# def _rects_close(a: fitz.Rect, b: fitz.Rect, gap: float) -> bool:
#     return not (a.x1+gap < b.x0 or b.x1+gap < a.x0 or
#                 a.y1+gap < b.y0 or b.y1+gap < a.y0)


# def _cluster_drawings(page: fitz.Page) -> List[fitz.Rect]:
#     """
#     Collect all vector drawing rects on the page, merge nearby ones into
#     figure clusters, filter junk (too small, wrong aspect ratio).
#     """
#     drawings = page.get_drawings()
#     if not drawings:
#         return []

#     rects = [fitz.Rect(d["rect"]) for d in drawings
#              if d.get("rect") and d["rect"].width > 1 and d["rect"].height > 1]
#     if not rects:
#         return []

#     # Iterative union-merge
#     clusters, changed = rects[:], True
#     while changed:
#         changed, merged, used = False, [], [False] * len(clusters)
#         for i in range(len(clusters)):
#             if used[i]:
#                 continue
#             cur = clusters[i]
#             for j in range(i + 1, len(clusters)):
#                 if used[j]:
#                     continue
#                 if _rects_close(cur, clusters[j], CLUSTER_GAP):
#                     cur, used[j], changed = _union_rect(cur, clusters[j]), True, True
#             merged.append(cur)
#         clusters = merged

#     result = []
#     for r in clusters:
#         if r.width * r.height < MIN_CLUSTER_AREA:
#             continue
#         if max(r.width, r.height) / max(min(r.width, r.height), 1) > CLUSTER_MAX_ASPECT:
#             continue
#         result.append(r)
#     return result


# def _pad_rect(r: fitz.Rect, pad: float, page_rect: fitz.Rect) -> fitz.Rect:
#     return fitz.Rect(
#         max(page_rect.x0, r.x0 - pad), max(page_rect.y0, r.y0 - pad),
#         min(page_rect.x1, r.x1 + pad), min(page_rect.y1, r.y1 + pad),
#     )


# # ═══════════════════════════════════════════════════════════════════════════════
# # CAPTION EXTRACTION — PyMuPDF text blocks
# # ═══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class _Caption:
#     text:  str
#     rect:  fitz.Rect    # PDF point space


# def _extract_captions(page: fitz.Page) -> List[_Caption]:
#     """
#     Extract caption candidates from page text blocks using PyMuPDF.
#     Uses two signals:
#       1. Text matching _CAPTION_RE ("Fig 3", "Figure 2:", etc.)
#       2. Font size — captions are typically smaller than body text
#          (we flag blocks whose font size is in the bottom 25% of page fonts)
#     Returns list of _Caption with PDF-point-space rects.
#     """
#     captions: List[_Caption] = []
#     blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

#     font_sizes = []
#     for block in blocks:
#         if block.get("type") != 0:   # 0 = text block
#             continue
#         for line in block.get("lines", []):
#             for span in line.get("spans", []):
#                 font_sizes.append(span.get("size", 12))

#     small_threshold = (
#         np.percentile(font_sizes, 25) if len(font_sizes) >= 4 else 0
#     )

#     for block in blocks:
#         if block.get("type") != 0:
#             continue
#         block_text = " ".join(
#             span["text"]
#             for line in block.get("lines", [])
#             for span in line.get("spans", [])
#         ).strip()

#         if not block_text:
#             continue

#         rect = fitz.Rect(block["bbox"])

#         # Primary signal: text pattern match
#         if _CAPTION_RE.search(block_text):
#             captions.append(_Caption(text=block_text, rect=rect))
#             continue

#         # Secondary signal: small font below a figure
#         avg_size = np.mean([
#             span.get("size", 12)
#             for line in block.get("lines", [])
#             for span in line.get("spans", [])
#         ]) if block.get("lines") else 12

#         if avg_size <= small_threshold and len(block_text) > 10:
#             captions.append(_Caption(text=block_text, rect=rect))

#     return captions


# def _match_caption(
#     fig_rect:  fitz.Rect,
#     captions:  List[_Caption],
# ) -> str:
#     """
#     Return the caption text closest to the figure (above or below),
#     within CAPTION_MAX_DIST_PT PDF points. Pure spatial — language agnostic.
#     """
#     if not captions:
#         return ""
#     fig_cy   = (fig_rect.y0 + fig_rect.y1) / 2
#     best, bd = "", float("inf")
#     for cap in captions:
#         # Distance from figure bottom to caption top (below) or
#         # caption bottom to figure top (above)
#         dist = min(
#             abs(cap.rect.y0 - fig_rect.y1),   # caption below figure
#             abs(fig_rect.y0 - cap.rect.y1),   # caption above figure
#         )
#         if dist < bd:
#             bd, best = dist, cap.text
#     return best if bd <= CAPTION_MAX_DIST_PT else ""


# # ═══════════════════════════════════════════════════════════════════════════════
# # XOBJECT PLACEMENT
# # ═══════════════════════════════════════════════════════════════════════════════

# def _xobject_rect_on_page(
#     page: fitz.Page,
#     xref: int,
# ) -> Optional[fitz.Rect]:
#     """
#     Find where xref is placed on the page (PDF point space).
#     Uses get_image_rects which returns all placement rects for a given xref.
#     """
#     try:
#         rects = page.get_image_rects(xref)
#         if rects:
#             return fitz.Rect(rects[0])
#     except Exception:
#         pass
#     return None


# def _vector_overlap_fraction(
#     xobj_rect:  fitz.Rect,
#     clusters:   List[fitz.Rect],
# ) -> float:
#     """Fraction of xobj_rect covered by vector drawing clusters."""
#     if not clusters or xobj_rect.is_empty:
#         return 0.0
#     area = xobj_rect.width * xobj_rect.height
#     if area <= 0:
#         return 0.0
#     covered = sum(
#         (xobj_rect & c).get_area()
#         for c in clusters
#         if not (xobj_rect & c).is_empty
#     )
#     return covered / area


# # ═══════════════════════════════════════════════════════════════════════════════
# # PER-PAGE TASK COLLECTION
# # ═══════════════════════════════════════════════════════════════════════════════

# def _collect_page_tasks(
#     page:        fitz.Page,
#     doc:         fitz.Document,
#     page_num:    int,
#     seen_xref:   Set[int],
#     seen_hashes: List[int],
# ) -> List[FigureTask]:
#     """
#     Run all 5 layers on one page. Returns FigureTask list.
#     Each layer is isolated in try/except — one failure ≠ page abort.
#     """
#     tasks:    List[FigureTask] = []
#     fig_idx:  int              = 1
#     captions: List[_Caption]   = []
#     page_rect = page.rect

#     try:
#         captions = _extract_captions(page)
#     except Exception as e:
#         logger.debug(f"Page {page_num}: caption extraction failed — {e}")

#     # Pre-compute drawing clusters once (used by Layers B and C)
#     drawing_clusters: List[fitz.Rect] = []
#     try:
#         drawing_clusters = _cluster_drawings(page)
#     except Exception as e:
#         logger.debug(f"Page {page_num}: drawing cluster failed — {e}")

#     layer_a_found: List[int] = []

#     # ══════════════════════════════════════════════════════════════════════════
#     # LAYER A — Embedded raster xobjects
#     # ══════════════════════════════════════════════════════════════════════════
#     try:
#         for xobj in page.get_images(full=True):
#             xref     = xobj[0]
#             smask    = xobj[1]
#             native_w = xobj[2]
#             native_h = xobj[3]

#             if xref in seen_xref:
#                 continue
#             if smask and xref == smask:          # alpha mask, not a figure
#                 continue
#             if native_w < MIN_XOBJ_WIDTH or native_h < MIN_XOBJ_HEIGHT:
#                 continue

#             try:
#                 raw = doc.extract_image(xref)
#             except Exception as e:
#                 logger.debug(f"Page {page_num}: extract_image({xref}) — {e}")
#                 continue

#             # ── LAYER C check: does this xobject have vector overlay? ─────────
#             xobj_rect   = _xobject_rect_on_page(page, xref)
#             use_render  = False

#             if xobj_rect and drawing_clusters:
#                 overlap = _vector_overlap_fraction(xobj_rect, drawing_clusters)
#                 if overlap > VECTOR_OVERLAY_THRESHOLD:
#                     # Render the page region — composites raster + vector overlay
#                     try:
#                         padded = _pad_rect(xobj_rect, CROP_PADDING, page_rect)
#                         pix    = _render_rect(page, padded)
#                         img    = _trim_whitespace(_pix_to_pil(pix))
#                         if _is_valid_figure(img) and not _is_duplicate(img, seen_hashes):
#                             seen_xref.add(xref)
#                             layer_a_found.append(xref)
#                             caption = _match_caption(xobj_rect, captions)
#                             label   = (
#                                 f"Page {page_num} — Figure {fig_idx} "
#                                 f"(raster+vector {img.width}×{img.height}px)"
#                             )
#                             tasks.append(FigureTask(
#                                 img=img, page_num=page_num, caption=caption,
#                                 label=label, fig_idx=fig_idx, layer="mixed",
#                             ))
#                             fig_idx  += 1
#                             use_render = True
#                     except Exception as e:
#                         logger.debug(f"Page {page_num}: mixed render failed — {e}")

#             if not use_render:
#                 img = _pil_from_bytes(raw["image"], raw.get("ext", ""))
#                 if img is None:
#                     continue
#                 img = _trim_whitespace(img)
#                 if not _is_valid_figure(img):
#                     continue
#                 if _is_duplicate(img, seen_hashes):
#                     logger.debug(f"Page {page_num}: xref {xref} duplicate, skipping")
#                     continue
#                 seen_xref.add(xref)
#                 layer_a_found.append(xref)
#                 caption = _match_caption(
#                     xobj_rect if xobj_rect else fitz.Rect(), captions
#                 )
#                 label = (
#                     f"Page {page_num} — Figure {fig_idx} "
#                     f"(raster {img.width}×{img.height}px)"
#                 )
#                 tasks.append(FigureTask(
#                     img=img, page_num=page_num, caption=caption,
#                     label=label, fig_idx=fig_idx, layer="raster",
#                 ))
#                 fig_idx += 1

#     except Exception as e:
#         logger.warning(f"Page {page_num}: Layer A failed — {e}")

#     # ══════════════════════════════════════════════════════════════════════════
#     # LAYER B — Vector drawing clusters
#     # Only runs when Layer A found nothing on this page
#     # ══════════════════════════════════════════════════════════════════════════
#     if not layer_a_found:
#         try:
#             page_text_len = len(page.get_text().strip())
#             is_scanned    = page_text_len < SCAN_TEXT_THRESHOLD

#             if not is_scanned:
#                 for cluster in drawing_clusters:
#                     try:
#                         padded = _pad_rect(cluster, CROP_PADDING, page_rect)
#                         pix    = _render_rect(page, padded)
#                         img    = _trim_whitespace(_pix_to_pil(pix))
#                         if not _is_valid_figure(img):
#                             continue
#                         if _is_duplicate(img, seen_hashes):
#                             continue
#                         caption = _match_caption(cluster, captions)
#                         label   = (
#                             f"Page {page_num} — Figure {fig_idx} "
#                             f"(vector {img.width}×{img.height}px, "
#                             f"bbox=[{cluster.x0:.0f},{cluster.y0:.0f},"
#                             f"{cluster.x1:.0f},{cluster.y1:.0f}]pt)"
#                         )
#                         tasks.append(FigureTask(
#                             img=img, page_num=page_num, caption=caption,
#                             label=label, fig_idx=fig_idx, layer="vector",
#                         ))
#                         fig_idx += 1
#                     except Exception as e:
#                         logger.debug(f"Page {page_num}: cluster render failed — {e}")

#         except Exception as e:
#             logger.warning(f"Page {page_num}: Layer B failed — {e}")

#     # ══════════════════════════════════════════════════════════════════════════
#     # LAYER E — Form XObjects (InDesign / Illustrator / PPT exports)
#     # Only runs when Layer A found nothing
#     # ══════════════════════════════════════════════════════════════════════════
#     if not layer_a_found:
#         try:
#             for xref, name in page.get_xobjects():
#                 if xref in seen_xref:
#                     continue
#                 try:
#                     subtype = doc.xref_get_key(xref, "Subtype")[1]
#                     if subtype != "/Form":
#                         continue
#                     rect = _xobject_rect_on_page(page, xref)
#                     if rect is None or rect.is_empty:
#                         continue
#                     padded = _pad_rect(rect, CROP_PADDING, page_rect)
#                     pix    = _render_rect(page, padded)
#                     img    = _trim_whitespace(_pix_to_pil(pix))
#                     if not _is_valid_figure(img):
#                         continue
#                     if _is_duplicate(img, seen_hashes):
#                         continue
#                     seen_xref.add(xref)
#                     caption = _match_caption(rect, captions)
#                     label   = (
#                         f"Page {page_num} — Figure {fig_idx} "
#                         f"(form-xobj/{name} {img.width}×{img.height}px)"
#                     )
#                     tasks.append(FigureTask(
#                         img=img, page_num=page_num, caption=caption,
#                         label=label, fig_idx=fig_idx, layer="form",
#                     ))
#                     fig_idx += 1
#                 except Exception as e:
#                     logger.debug(f"Page {page_num}: form xobj {name} failed — {e}")
#         except Exception as e:
#             logger.warning(f"Page {page_num}: Layer E failed — {e}")

#     # ══════════════════════════════════════════════════════════════════════════
#     # LAYER D — Scanned / image-only full-page fallback
#     # Only runs when ALL other layers produced nothing
#     # ══════════════════════════════════════════════════════════════════════════
#     if not tasks:
#         try:
#             page_text_len = len(page.get_text().strip())
#             if page_text_len < SCAN_TEXT_THRESHOLD:
#                 pix = _render_full_page(page)
#                 img = _trim_whitespace(_pix_to_pil(pix))
#                 if _is_valid_figure(img) and not _is_duplicate(img, seen_hashes):
#                     label = (
#                         f"Page {page_num} — Figure {fig_idx} "
#                         f"(scanned {img.width}×{img.height}px)"
#                     )
#                     tasks.append(FigureTask(
#                         img=img, page_num=page_num,
#                         caption=" ".join(c.text for c in captions[:2]),
#                         label=label, fig_idx=fig_idx, layer="scanned",
#                     ))
#         except Exception as e:
#             logger.warning(f"Page {page_num}: Layer D failed — {e}")

#     return tasks


# # ═══════════════════════════════════════════════════════════════════════════════
# # VISION DESCRIPTION
# # ═══════════════════════════════════════════════════════════════════════════════

# _EQ_HINT = (
#     "This figure likely contains mathematical equations or notation. "
#     "Transcribe visible formulas using LaTeX notation where possible and "
#     "explain what each formula represents conceptually.\n"
# )


# def _describe_figure(task: FigureTask) -> Optional[str]:
#     try:
#         img = task.img
#         b64 = _to_base64_jpeg(_resize_for_vision(img))

#         # Equation detection: low colour diversity = likely math/text figure
#         arr      = np.array(img.convert("L"))
#         eq_hint  = _EQ_HINT if (len(np.unique(arr)) / 256.0) < 0.15 else ""
#         cap_hint = f'\nFigure caption: "{task.caption}"\n' if task.caption else ""

#         prompt = (
#             f"You are analyzing a figure from page {task.page_num} of a PDF."
#             f"{cap_hint}{eq_hint}\n"
#             "Respond in EXACTLY this structure:\n\n"
#             "FIGURE TYPE: <bar chart | line chart | scatter plot | heatmap | "
#             "flowchart | architecture diagram | attention map | network diagram | "
#             "photograph | screenshot | infographic | table | equation | other>\n"
#             "KEY FINDING: <one sentence — the single most important insight>\n"
#             "DESCRIPTION: <thorough description — include all visible labels, "
#             "axis names and ranges, legend entries, color meanings, annotated "
#             "regions, component names, data trends, and what every visual "
#             "element represents. Detailed enough that someone without the "
#             "image can fully answer questions about it.>\n"
#             "SEARCHABLE TERMS: <comma-separated domain terms, metric names, "
#             "model names, dataset names, acronyms visible in the figure>"
#         )

#         response = _openai.chat.completions.create(
#             model=VISION_MODEL,
#             max_tokens=VISION_MAX_TOKENS,
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url":    f"data:image/jpeg;base64,{b64}",
#                             "detail": VISION_DETAIL,
#                         },
#                     },
#                 ],
#             }],
#         )
#         return response.choices[0].message.content.strip()

#     except Exception as e:
#         logger.warning(f"Vision API failed — {task.label}: {e}")
#         return None


# # ═══════════════════════════════════════════════════════════════════════════════
# # IMAGE EXTRACTION ORCHESTRATOR
# # ═══════════════════════════════════════════════════════════════════════════════

# def _extract_image_docs(file_bytes: bytes, thread_id: str) -> List[Document]:
#     """
#     Walk every page through all 5 layers, collect FigureTasks, then run
#     vision + S3 upload in parallel.
#     """
#     doc = _open_pdf(file_bytes)
#     if doc is None:
#         return []

#     seen_xref:   Set[int]  = set()
#     seen_hashes: List[int] = []
#     all_tasks:   List[FigureTask] = []

#     # ── Pass 1: collect tasks (sequential — page ordering matters for dedup) ──
#     for page_num, page in enumerate(doc, start=1):
#         try:
#             page_tasks = _collect_page_tasks(
#                 page, doc, page_num, seen_xref, seen_hashes
#             )
#             all_tasks.extend(page_tasks)
#             if page_tasks:
#                 logger.debug(
#                     f"Page {page_num}: {len(page_tasks)} figures "
#                     f"({', '.join(t.layer for t in page_tasks)})"
#                 )
#         except Exception as e:
#             logger.warning(f"Page {page_num}: task collection failed — {e}")

#     doc.close()
#     logger.info(f"Images: {len(all_tasks)} figures queued for vision")

#     if not all_tasks:
#         return []

#     # ── Pass 2: parallel vision + S3 upload ──────────────────────────────────
#     def _run_task(task: FigureTask) -> Optional[Document]:
#         description = _describe_figure(task)
#         if not description:
#             return None

#         content_parts = [f"[{task.label}]"]
#         if task.caption:
#             content_parts.append(f"Caption: {task.caption}")
#         content_parts.append(description)

#         try:
#             image_url = upload_page_image(
#                 task.img, thread_id, task.page_num, task.fig_idx
#             )
#         except Exception as e:
#             logger.warning(f"S3 upload failed — {task.label}: {e}")
#             return None

#         return Document(
#             page_content="\n".join(content_parts),
#             metadata={
#                 "thread_id":    thread_id,
#                 "content_type": "image",
#                 "page":         task.page_num,
#                 "figure_index": task.fig_idx,
#                 "source":       "pdf",
#                 "caption":      task.caption,
#                 "layer":        task.layer,
#                 "image_ref":    image_url,
#             },
#         )

#     image_docs: List[Document] = []
#     with ThreadPoolExecutor(max_workers=VISION_WORKERS) as executor:
#         future_map = {executor.submit(_run_task, t): t for t in all_tasks}
#         for future in as_completed(future_map):
#             task = future_map[future]
#             try:
#                 result = future.result()
#             except Exception as e:
#                 logger.warning(f"Task error — {task.label}: {e}")
#                 continue
#             if result:
#                 image_docs.append(result)
#                 logger.info(f"  ✅ {task.label} [{task.layer}]")

#     logger.info(f"Images: {len(image_docs)} documents created")
#     return image_docs


# # ═══════════════════════════════════════════════════════════════════════════════
# # TEXT EXTRACTION
# # ═══════════════════════════════════════════════════════════════════════════════

# def _extract_text_chunks(file_bytes: bytes, thread_id: str) -> List[Document]:
#     """
#     Extract selectable text using PyMuPDF page.get_text("blocks").
#     Skips pages with < SCAN_TEXT_THRESHOLD chars (scanned pages).
#     Preserves reading order via block sort (top→bottom, left→right).
#     """
#     doc = _open_pdf(file_bytes)
#     if doc is None:
#         return []

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=TEXT_CHUNK_SIZE,
#         chunk_overlap=TEXT_CHUNK_OVERLAP,
#     )
#     all_docs: List[Document] = []

#     for page_num, page in enumerate(doc, start=1):
#         try:
#             # Sort blocks top→bottom then left→right for correct reading order
#             blocks = sorted(
#                 page.get_text("blocks"),
#                 key=lambda b: (round(b[1] / 10) * 10, b[0]),
#             )
#             page_text = "\n".join(
#                 b[4].strip() for b in blocks
#                 if b[6] == 0 and b[4].strip()   # b[6]==0 → text block
#             )
#             if len(page_text) < SCAN_TEXT_THRESHOLD:
#                 continue   # scanned page — no usable text
#             for chunk in splitter.split_text(page_text):
#                 all_docs.append(Document(
#                     page_content=chunk,
#                     metadata={
#                         "thread_id":    thread_id,
#                         "content_type": "text",
#                         "page":         page_num,
#                         "source":       "pdf",
#                     },
#                 ))
#         except Exception as e:
#             logger.warning(f"Page {page_num}: text extraction failed — {e}")

#     doc.close()
#     logger.info(f"Text: {len(all_docs)} chunks")
#     return all_docs


# # ═══════════════════════════════════════════════════════════════════════════════
# # TABLE EXTRACTION — pdfplumber
# # ═══════════════════════════════════════════════════════════════════════════════

# def _list_to_markdown(table: List[List]) -> str:
#     if not table or len(table) < MIN_TABLE_ROWS:
#         return ""
#     cleaned = [[str(c).strip() if c else "" for c in row] for row in table]
#     if len(cleaned[0]) < MIN_TABLE_COLS:
#         return ""
#     hdr  = cleaned[0]
#     sep  = ["---"] * len(hdr)
#     lines = [
#         "| " + " | ".join(hdr) + " |",
#         "| " + " | ".join(sep) + " |",
#     ]
#     for row in cleaned[1:]:
#         p = row + [""] * (len(hdr) - len(row))
#         lines.append("| " + " | ".join(p[:len(hdr)]) + " |")
#     return "\n".join(lines)


# def _extract_table_docs(file_bytes: bytes, thread_id: str) -> List[Document]:
#     """pdfplumber table extraction with per-page error isolation."""
#     docs: List[Document] = []
#     try:
#         with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
#             for page_num, page in enumerate(pdf.pages, start=1):
#                 try:
#                     for idx, table in enumerate(page.extract_tables() or [], start=1):
#                         md = _list_to_markdown(table)
#                         if not md:
#                             continue
#                         docs.append(Document(
#                             page_content=f"[Page {page_num} — Table {idx}]\n{md}",
#                             metadata={
#                                 "thread_id":    thread_id,
#                                 "content_type": "table",
#                                 "page":         page_num,
#                                 "table_index":  idx,
#                                 "source":       "pdf",
#                             },
#                         ))
#                 except Exception as e:
#                     logger.warning(f"Page {page_num}: table extraction failed — {e}")
#     except Exception as e:
#         logger.warning(f"pdfplumber failed: {e}")

#     logger.info(f"Tables: {len(docs)} tables extracted")
#     return docs


# # ═══════════════════════════════════════════════════════════════════════════════
# # MAIN ENTRY POINT
# # ═══════════════════════════════════════════════════════════════════════════════

# def ingest_pdf_multimodal(
#     file_bytes: bytes,
#     thread_id:  str,
#     *,
#     extract_text:   bool = True,
#     extract_images: bool = True,
#     extract_tables: bool = True,
# ) -> dict:
#     all_docs: List[Document] = []
#     counts = {"text": 0, "images": 0, "tables": 0}

#     if extract_text:
#         docs = _extract_text_chunks(file_bytes, thread_id)
#         all_docs.extend(docs)
#         counts["text"] = len(docs)

#     if extract_tables:
#         docs = _extract_table_docs(file_bytes, thread_id)
#         all_docs.extend(docs)
#         counts["tables"] = len(docs)

#     if extract_images:
#         docs = _extract_image_docs(file_bytes, thread_id)
#         all_docs.extend(docs)
#         counts["images"] = len(docs)

#     if not all_docs:
#         logger.warning(f"No content extracted from PDF for thread={thread_id}")
#         return {**counts, "total": 0}

#     cohere_vector_store.add_texts(
#         texts=[d.page_content for d in all_docs],
#         metadatas=[d.metadata for d in all_docs],
#     )

#     counts["total"] = len(all_docs)
#     logger.info(
#         f"✅ Multimodal ingestion complete — thread={thread_id} | "
#         f"text={counts['text']} table={counts['tables']} "
#         f"image={counts['images']} total={counts['total']} | "
#         f"store=crag_multimodal (Cohere)"
#     )
#     return counts


# def ingest_pdf(file_bytes: bytes, thread_id: str) -> None:
#     ingest_pdf_multimodal(file_bytes, thread_id)

"""
pdf_ingest.py — Universal multimodal PDF ingestion using PyMuPDF only.

... (rest of the docstring unchanged) ...
"""

import io
import base64
import logging
import os
import re
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

from backend.rag.cohere_store import cohere_vector_store
# from rag.cohere_store import cohere_vector_store
from backend.ingest.image_store import upload_page_image
# from ingest.image_store import upload_page_image

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

# ── Layer A: raster xobject filters ──────────────────────────────────────────
MIN_XOBJ_WIDTH  = 80      # px — skip tiny icons / bullets
MIN_XOBJ_HEIGHT = 80      # px

# ── Layer B: vector cluster tuning ────────────────────────────────────────────
CLUSTER_GAP      = 40     # pt — rects within this distance are merged
MIN_CLUSTER_AREA = 2500   # pt² (50×50) — minimum qualifying cluster
CLUSTER_MAX_ASPECT = 10.0 # skip long thin rules (table borders, dividers)
CROP_PADDING     = 8      # pt — padding around cluster rect before render

# ── Layer C: mixed page detection ─────────────────────────────────────────────
VECTOR_OVERLAY_THRESHOLD = 0.12

# ── Layer D: scanned page detection ───────────────────────────────────────────
SCAN_TEXT_THRESHOLD = 50  # chars — below this = likely scanned page

# ── Figure quality filters (all layers) ──────────────────────────────────────
MIN_FIGURE_AREA   = 80 * 80   # px² after crop + trim
MIN_CONTENT_RATIO = 0.02      # fraction non-white pixels
MAX_FIGURE_ASPECT = 8.0       # skip extreme horizontal/vertical strips

# ── Duplicate suppression ─────────────────────────────────────────────────────
DEDUP_HASH_DISTANCE = 8       # max perceptual hash Hamming distance

# ── Caption matching ──────────────────────────────────────────────────────────
CAPTION_MAX_DIST_PT = 80      # pt — max vertical gap figure↔caption in PDF space

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
    layer:      str           # "raster"|"vector"|"mixed"|"scanned"|"form"


# ═══════════════════════════════════════════════════════════════════════════════
# PDF OPEN — encryption handling
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
# IMAGE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _pil_from_bytes(raw: bytes, ext: str = "") -> Optional[Image.Image]:
    """Decode raw image bytes → PIL RGB. Returns None on failure."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img.convert("RGB")
    except Exception as e:
        logger.debug(f"PIL decode failed (ext={ext}): {e}")
        return None


def _pix_to_pil(pix: fitz.Pixmap) -> Image.Image:
    """Convert fitz Pixmap → PIL RGB Image."""
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _render_rect(
    page: fitz.Page,
    rect: fitz.Rect,
) -> fitz.Pixmap:
    """Render a clipped rect of the page at RENDER_DPI."""
    scale = RENDER_DPI / 72
    mat   = fitz.Matrix(scale, scale)
    return page.get_pixmap(matrix=mat, clip=rect, alpha=False)


def _render_full_page(page: fitz.Page) -> fitz.Pixmap:
    """Render entire page at RENDER_DPI."""
    scale = RENDER_DPI / 72
    mat   = fitz.Matrix(scale, scale)
    return page.get_pixmap(matrix=mat, alpha=False)


def _content_ratio(img: Image.Image) -> float:
    """Fraction of non-white pixels."""
    arr = np.array(img.convert("L"))
    return float((arr < 245).sum()) / max(arr.size, 1)


def _trim_whitespace(img: Image.Image) -> Image.Image:
    """Remove near-white border rows/cols."""
    arr  = np.array(img.convert("L"))
    mask = arr < 245
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return img
    return img.crop((int(cols[0]), int(rows[0]),
                     int(cols[-1]) + 1, int(rows[-1]) + 1))


def _is_valid_figure(img: Image.Image) -> bool:
    """Passes minimum area, content ratio, and aspect ratio filters."""
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


# ── Perceptual hash (pure numpy — no external lib) ───────────────────────────

def _phash(img: Image.Image) -> int:
    """8×8 average perceptual hash as integer."""
    small = np.array(img.convert("L").resize((8, 8), Image.LANCZOS), dtype=float)
    bits  = (small >= small.mean()).flatten()
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def _is_duplicate(img: Image.Image, seen: List[int]) -> bool:
    """True if img is visually similar to any previously seen figure."""
    h = _phash(img)
    for prev in seen:
        if bin(h ^ prev).count("1") <= DEDUP_HASH_DISTANCE:
            return True
    seen.append(h)
    return False


# ── Local saving ──────────────────────────────────────────────────────────────

def _save_image_locally(img: Image.Image, thread_id: str, page_num: int, fig_idx: int) -> Optional[Path]:
    """Save image to local folder. Returns path if saved, None otherwise."""
    if not SAVE_IMAGES_LOCAL:
        return None
    # Create thread‑specific subdirectory
    thread_dir = LOCAL_IMAGE_DIR / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename (same pattern as S3)
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    filename = f"page_{page_num:03d}_fig_{fig_idx}_{unique_id}.jpg"
    filepath = thread_dir / filename

    # Save as JPEG (same quality as S3 upload)
    try:
        img.convert("RGB").save(filepath, "JPEG", quality=88)
        logger.debug(f"Saved locally: {filepath}")
        return filepath
    except Exception as e:
        logger.warning(f"Failed to save image locally: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR CLUSTER EXTRACTION (Layer B core)
# ═══════════════════════════════════════════════════════════════════════════════

def _union_rect(a: fitz.Rect, b: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(min(a.x0,b.x0), min(a.y0,b.y0),
                     max(a.x1,b.x1), max(a.y1,b.y1))


def _rects_close(a: fitz.Rect, b: fitz.Rect, gap: float) -> bool:
    return not (a.x1+gap < b.x0 or b.x1+gap < a.x0 or
                a.y1+gap < b.y0 or b.y1+gap < a.y0)


def _cluster_drawings(page: fitz.Page) -> List[fitz.Rect]:
    """
    Collect all vector drawing rects on the page, merge nearby ones into
    figure clusters, filter junk (too small, wrong aspect ratio).
    """
    drawings = page.get_drawings()
    if not drawings:
        return []

    rects = [fitz.Rect(d["rect"]) for d in drawings
             if d.get("rect") and d["rect"].width > 1 and d["rect"].height > 1]
    if not rects:
        return []

    # Iterative union-merge
    clusters, changed = rects[:], True
    while changed:
        changed, merged, used = False, [], [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            cur = clusters[i]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if _rects_close(cur, clusters[j], CLUSTER_GAP):
                    cur, used[j], changed = _union_rect(cur, clusters[j]), True, True
            merged.append(cur)
        clusters = merged

    result = []
    for r in clusters:
        if r.width * r.height < MIN_CLUSTER_AREA:
            continue
        if max(r.width, r.height) / max(min(r.width, r.height), 1) > CLUSTER_MAX_ASPECT:
            continue
        result.append(r)
    return result


def _pad_rect(r: fitz.Rect, pad: float, page_rect: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(
        max(page_rect.x0, r.x0 - pad), max(page_rect.y0, r.y0 - pad),
        min(page_rect.x1, r.x1 + pad), min(page_rect.y1, r.y1 + pad),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CAPTION EXTRACTION — PyMuPDF text blocks
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Caption:
    text:  str
    rect:  fitz.Rect    # PDF point space


def _extract_captions(page: fitz.Page) -> List[_Caption]:
    """
    Extract caption candidates from page text blocks using PyMuPDF.
    Uses two signals:
      1. Text matching _CAPTION_RE ("Fig 3", "Figure 2:", etc.)
      2. Font size — captions are typically smaller than body text
         (we flag blocks whose font size is in the bottom 25% of page fonts)
    Returns list of _Caption with PDF-point-space rects.
    """
    captions: List[_Caption] = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    font_sizes = []
    for block in blocks:
        if block.get("type") != 0:   # 0 = text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_sizes.append(span.get("size", 12))

    small_threshold = (
        np.percentile(font_sizes, 25) if len(font_sizes) >= 4 else 0
    )

    for block in blocks:
        if block.get("type") != 0:
            continue
        block_text = " ".join(
            span["text"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()

        if not block_text:
            continue

        rect = fitz.Rect(block["bbox"])

        # Primary signal: text pattern match
        if _CAPTION_RE.search(block_text):
            captions.append(_Caption(text=block_text, rect=rect))
            continue

        # Secondary signal: small font below a figure
        avg_size = np.mean([
            span.get("size", 12)
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]) if block.get("lines") else 12

        if avg_size <= small_threshold and len(block_text) > 10:
            captions.append(_Caption(text=block_text, rect=rect))

    return captions


def _match_caption(
    fig_rect:  fitz.Rect,
    captions:  List[_Caption],
) -> str:
    """
    Return the caption text closest to the figure (above or below),
    within CAPTION_MAX_DIST_PT PDF points. Pure spatial — language agnostic.
    """
    if not captions:
        return ""
    fig_cy   = (fig_rect.y0 + fig_rect.y1) / 2
    best, bd = "", float("inf")
    for cap in captions:
        # Distance from figure bottom to caption top (below) or
        # caption bottom to figure top (above)
        dist = min(
            abs(cap.rect.y0 - fig_rect.y1),   # caption below figure
            abs(fig_rect.y0 - cap.rect.y1),   # caption above figure
        )
        if dist < bd:
            bd, best = dist, cap.text
    return best if bd <= CAPTION_MAX_DIST_PT else ""


# ═══════════════════════════════════════════════════════════════════════════════
# XOBJECT PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _xobject_rect_on_page(
    page: fitz.Page,
    xref: int,
) -> Optional[fitz.Rect]:
    """
    Find where xref is placed on the page (PDF point space).
    Uses get_image_rects which returns all placement rects for a given xref.
    """
    try:
        rects = page.get_image_rects(xref)
        if rects:
            return fitz.Rect(rects[0])
    except Exception:
        pass
    return None


def _vector_overlap_fraction(
    xobj_rect:  fitz.Rect,
    clusters:   List[fitz.Rect],
) -> float:
    """Fraction of xobj_rect covered by vector drawing clusters."""
    if not clusters or xobj_rect.is_empty:
        return 0.0
    area = xobj_rect.width * xobj_rect.height
    if area <= 0:
        return 0.0
    covered = sum(
        (xobj_rect & c).get_area()
        for c in clusters
        if not (xobj_rect & c).is_empty
    )
    return covered / area


# ═══════════════════════════════════════════════════════════════════════════════
# PER-PAGE TASK COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_page_tasks(
    page:        fitz.Page,
    doc:         fitz.Document,
    page_num:    int,
    seen_xref:   Set[int],
    seen_hashes: List[int],
) -> List[FigureTask]:
    """
    Run all 5 layers on one page. Returns FigureTask list.
    Each layer is isolated in try/except — one failure ≠ page abort.
    """
    tasks:    List[FigureTask] = []
    fig_idx:  int              = 1
    captions: List[_Caption]   = []
    page_rect = page.rect

    try:
        captions = _extract_captions(page)
    except Exception as e:
        logger.debug(f"Page {page_num}: caption extraction failed — {e}")

    # Pre-compute drawing clusters once (used by Layers B and C)
    drawing_clusters: List[fitz.Rect] = []
    try:
        drawing_clusters = _cluster_drawings(page)
    except Exception as e:
        logger.debug(f"Page {page_num}: drawing cluster failed — {e}")

    layer_a_found: List[int] = []

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER A — Embedded raster xobjects
    # ══════════════════════════════════════════════════════════════════════════
    try:
        for xobj in page.get_images(full=True):
            xref     = xobj[0]
            smask    = xobj[1]
            native_w = xobj[2]
            native_h = xobj[3]

            if xref in seen_xref:
                continue
            if smask and xref == smask:          # alpha mask, not a figure
                continue
            if native_w < MIN_XOBJ_WIDTH or native_h < MIN_XOBJ_HEIGHT:
                continue

            try:
                raw = doc.extract_image(xref)
            except Exception as e:
                logger.debug(f"Page {page_num}: extract_image({xref}) — {e}")
                continue

            # ── LAYER C check: does this xobject have vector overlay? ─────────
            xobj_rect   = _xobject_rect_on_page(page, xref)
            use_render  = False

            if xobj_rect and drawing_clusters:
                overlap = _vector_overlap_fraction(xobj_rect, drawing_clusters)
                if overlap > VECTOR_OVERLAY_THRESHOLD:
                    # Render the page region — composites raster + vector overlay
                    try:
                        padded = _pad_rect(xobj_rect, CROP_PADDING, page_rect)
                        pix    = _render_rect(page, padded)
                        img    = _trim_whitespace(_pix_to_pil(pix))
                        if _is_valid_figure(img) and not _is_duplicate(img, seen_hashes):
                            seen_xref.add(xref)
                            layer_a_found.append(xref)
                            caption = _match_caption(xobj_rect, captions)
                            label   = (
                                f"Page {page_num} — Figure {fig_idx} "
                                f"(raster+vector {img.width}×{img.height}px)"
                            )
                            tasks.append(FigureTask(
                                img=img, page_num=page_num, caption=caption,
                                label=label, fig_idx=fig_idx, layer="mixed",
                            ))
                            fig_idx  += 1
                            use_render = True
                    except Exception as e:
                        logger.debug(f"Page {page_num}: mixed render failed — {e}")

            if not use_render:
                img = _pil_from_bytes(raw["image"], raw.get("ext", ""))
                if img is None:
                    continue
                img = _trim_whitespace(img)
                if not _is_valid_figure(img):
                    continue
                if _is_duplicate(img, seen_hashes):
                    logger.debug(f"Page {page_num}: xref {xref} duplicate, skipping")
                    continue
                seen_xref.add(xref)
                layer_a_found.append(xref)
                caption = _match_caption(
                    xobj_rect if xobj_rect else fitz.Rect(), captions
                )
                label = (
                    f"Page {page_num} — Figure {fig_idx} "
                    f"(raster {img.width}×{img.height}px)"
                )
                tasks.append(FigureTask(
                    img=img, page_num=page_num, caption=caption,
                    label=label, fig_idx=fig_idx, layer="raster",
                ))
                fig_idx += 1

    except Exception as e:
        logger.warning(f"Page {page_num}: Layer A failed — {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER B — Vector drawing clusters
    # Only runs when Layer A found nothing on this page
    # ══════════════════════════════════════════════════════════════════════════
    if not layer_a_found:
        try:
            page_text_len = len(page.get_text().strip())
            is_scanned    = page_text_len < SCAN_TEXT_THRESHOLD

            if not is_scanned:
                for cluster in drawing_clusters:
                    try:
                        padded = _pad_rect(cluster, CROP_PADDING, page_rect)
                        pix    = _render_rect(page, padded)
                        img    = _trim_whitespace(_pix_to_pil(pix))
                        if not _is_valid_figure(img):
                            continue
                        if _is_duplicate(img, seen_hashes):
                            continue
                        caption = _match_caption(cluster, captions)
                        label   = (
                            f"Page {page_num} — Figure {fig_idx} "
                            f"(vector {img.width}×{img.height}px, "
                            f"bbox=[{cluster.x0:.0f},{cluster.y0:.0f},"
                            f"{cluster.x1:.0f},{cluster.y1:.0f}]pt)"
                        )
                        tasks.append(FigureTask(
                            img=img, page_num=page_num, caption=caption,
                            label=label, fig_idx=fig_idx, layer="vector",
                        ))
                        fig_idx += 1
                    except Exception as e:
                        logger.debug(f"Page {page_num}: cluster render failed — {e}")

        except Exception as e:
            logger.warning(f"Page {page_num}: Layer B failed — {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER E — Form XObjects (InDesign / Illustrator / PPT exports)
    # Only runs when Layer A found nothing
    # ══════════════════════════════════════════════════════════════════════════
    if not layer_a_found:
        try:
            for xref, name in page.get_xobjects():
                if xref in seen_xref:
                    continue
                try:
                    subtype = doc.xref_get_key(xref, "Subtype")[1]
                    if subtype != "/Form":
                        continue
                    rect = _xobject_rect_on_page(page, xref)
                    if rect is None or rect.is_empty:
                        continue
                    padded = _pad_rect(rect, CROP_PADDING, page_rect)
                    pix    = _render_rect(page, padded)
                    img    = _trim_whitespace(_pix_to_pil(pix))
                    if not _is_valid_figure(img):
                        continue
                    if _is_duplicate(img, seen_hashes):
                        continue
                    seen_xref.add(xref)
                    caption = _match_caption(rect, captions)
                    label   = (
                        f"Page {page_num} — Figure {fig_idx} "
                        f"(form-xobj/{name} {img.width}×{img.height}px)"
                    )
                    tasks.append(FigureTask(
                        img=img, page_num=page_num, caption=caption,
                        label=label, fig_idx=fig_idx, layer="form",
                    ))
                    fig_idx += 1
                except Exception as e:
                    logger.debug(f"Page {page_num}: form xobj {name} failed — {e}")
        except Exception as e:
            logger.warning(f"Page {page_num}: Layer E failed — {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER D — Scanned / image-only full-page fallback
    # Only runs when ALL other layers produced nothing
    # ══════════════════════════════════════════════════════════════════════════
    if not tasks:
        try:
            page_text_len = len(page.get_text().strip())
            if page_text_len < SCAN_TEXT_THRESHOLD:
                pix = _render_full_page(page)
                img = _trim_whitespace(_pix_to_pil(pix))
                if _is_valid_figure(img) and not _is_duplicate(img, seen_hashes):
                    label = (
                        f"Page {page_num} — Figure {fig_idx} "
                        f"(scanned {img.width}×{img.height}px)"
                    )
                    tasks.append(FigureTask(
                        img=img, page_num=page_num,
                        caption=" ".join(c.text for c in captions[:2]),
                        label=label, fig_idx=fig_idx, layer="scanned",
                    ))
        except Exception as e:
            logger.warning(f"Page {page_num}: Layer D failed — {e}")

    return tasks


# ═══════════════════════════════════════════════════════════════════════════════
# VISION DESCRIPTION
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

        # Equation detection: low colour diversity = likely math/text figure
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
# IMAGE EXTRACTION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_image_docs(file_bytes: bytes, thread_id: str) -> List[Document]:
    """
    Walk every page through all 5 layers, collect FigureTasks, then run
    vision + S3 upload in parallel.
    """
    doc = _open_pdf(file_bytes)
    if doc is None:
        return []

    seen_xref:   Set[int]  = set()
    seen_hashes: List[int] = []
    all_tasks:   List[FigureTask] = []

    # ── Pass 1: collect tasks (sequential — page ordering matters for dedup) ──
    for page_num, page in enumerate(doc, start=1):
        try:
            page_tasks = _collect_page_tasks(
                page, doc, page_num, seen_xref, seen_hashes
            )
            all_tasks.extend(page_tasks)
            if page_tasks:
                logger.debug(
                    f"Page {page_num}: {len(page_tasks)} figures "
                    f"({', '.join(t.layer for t in page_tasks)})"
                )
        except Exception as e:
            logger.warning(f"Page {page_num}: task collection failed — {e}")

    doc.close()
    logger.info(f"Images: {len(all_tasks)} figures queued for vision")

    if not all_tasks:
        return []

    # ── Pass 2: parallel vision + S3 upload + optional local save ─────────────
    def _run_task(task: FigureTask) -> Optional[Document]:
        description = _describe_figure(task)
        if not description:
            return None

        content_parts = [f"[{task.label}]"]
        if task.caption:
            content_parts.append(f"Caption: {task.caption}")
        content_parts.append(description)

        # Upload to S3
        try:
            image_url = upload_page_image(
                task.img, thread_id, task.page_num, task.fig_idx
            )
        except Exception as e:
            logger.warning(f"S3 upload failed — {task.label}: {e}")
            image_url = None

        # Save locally if enabled
        if SAVE_IMAGES_LOCAL:
            _save_image_locally(task.img, thread_id, task.page_num, task.fig_idx)

        # If both upload and local save failed, we might still return a document without image_ref
        if image_url is None:
            logger.warning(f"No image URL for {task.label} — document will lack image_ref")
            # Still create document with description only
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
                    # image_ref omitted
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
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text_chunks(file_bytes: bytes, thread_id: str) -> List[Document]:
    """
    Extract selectable text using PyMuPDF page.get_text("blocks").
    Skips pages with < SCAN_TEXT_THRESHOLD chars (scanned pages).
    Preserves reading order via block sort (top→bottom, left→right).
    """
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
            # Sort blocks top→bottom then left→right for correct reading order
            blocks = sorted(
                page.get_text("blocks"),
                key=lambda b: (round(b[1] / 10) * 10, b[0]),
            )
            page_text = "\n".join(
                b[4].strip() for b in blocks
                if b[6] == 0 and b[4].strip()   # b[6]==0 → text block
            )
            if len(page_text) < SCAN_TEXT_THRESHOLD:
                continue   # scanned page — no usable text
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
# TABLE EXTRACTION — pdfplumber
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
    """pdfplumber table extraction with per-page error isolation."""
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
# MAIN ENTRY POINT
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

