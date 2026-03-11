# # """
# # image_store.py — Cloud image storage abstraction for CRAG pipeline.

# # Supports AWS S3 backend (IMAGE_STORE_BACKEND=s3).

# # .env keys:
# #     IMAGE_STORE_BACKEND=s3
# #     AWS_S3_BUCKET=your-bucket-name
# #     AWS_REGION=us-east-1
# #     AWS_ACCESS_KEY_ID=...
# #     AWS_SECRET_ACCESS_KEY=...
# #     AWS_S3_PUBLIC_BASE_URL=https://your-bucket.s3.amazonaws.com  
# # """

# # import io
# # import os
# # import logging
# # import uuid
# # from typing import Optional

# # from PIL import Image
# # from dotenv import load_dotenv

# # load_dotenv()

# # logger = logging.getLogger(__name__)

# # BACKEND          = os.getenv("IMAGE_STORE_BACKEND", "s3").lower()
# # THUMBNAIL_MAX_SIDE = 800
# # THUMBNAIL_QUALITY  = 75


# # def _image_to_bytes(img: Image.Image) -> bytes:
# #     """Resize and encode PIL Image as JPEG bytes."""
# #     w, h = img.size
# #     if max(w, h) > THUMBNAIL_MAX_SIDE:
# #         scale = THUMBNAIL_MAX_SIDE / max(w, h)
# #         img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
# #     buf = io.BytesIO()
# #     img.convert("RGB").save(buf, format="JPEG", quality=THUMBNAIL_QUALITY)
# #     return buf.getvalue()


# # def _upload_s3(image_bytes: bytes, key: str) -> str:
# #     try:
# #         import boto3
# #     except ImportError:
# #         raise RuntimeError("boto3 not installed. Run: pip install boto3")

# #     bucket   = os.getenv("AWS_S3_BUCKET")
# #     region   = os.getenv("AWS_REGION", "us-east-1")
# #     base_url = os.getenv("AWS_S3_PUBLIC_BASE_URL", f"https://{bucket}.s3.{region}.amazonaws.com")

# #     if not bucket:
# #         raise ValueError("AWS_S3_BUCKET env var not set")

# #     s3 = boto3.client(
# #         "s3",
# #         region_name=region,
# #         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
# #         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
# #     )
# #     s3.put_object(
# #         Bucket=bucket,
# #         Key=key,
# #         Body=image_bytes,
# #         ContentType="image/jpeg",
# #         # ACL removed — use bucket policy for public read on crag/* prefix
# #     )
# #     url = f"{base_url.rstrip('/')}/{key}"
# #     logger.info(f"S3 upload: s3://{bucket}/{key} → {url}")
# #     return url


# # def upload_page_image(
# #     img: Image.Image,
# #     thread_id: str,
# #     page_num: int,
# #     fig_idx: Optional[int] = None,   # NEW — figure index within the page
# # ) -> str:
# #     """
# #     Resize + upload a rendered image (full page or cropped figure) to S3.

# #     With unstructured extraction, fig_idx is the figure index on the page
# #     (1-based). The S3 key includes it so multiple figures on the same page
# #     get distinct keys:
# #         crag/{thread_id}/page_004_fig_2_a3f9c1b2.jpg   ← with fig_idx
# #         crag/{thread_id}/page_004_a3f9c1b2.jpg          ← fallback (no fig_idx)
# #     """
# #     image_bytes = _image_to_bytes(img)
# #     unique_id   = uuid.uuid4().hex[:8]

# #     if fig_idx is not None:
# #         filename = f"crag/{thread_id}/page_{page_num:03d}_fig_{fig_idx}_{unique_id}.jpg"
# #     else:
# #         filename = f"crag/{thread_id}/page_{page_num:03d}_{unique_id}.jpg"

# #     if BACKEND == "s3":
# #         return _upload_s3(image_bytes, filename)
# #     else:
# #         raise ValueError(f"Unknown IMAGE_STORE_BACKEND='{BACKEND}'. Use 's3'.")


# # def fetch_image_as_base64(url: str) -> Optional[str]:
# #     """
# #     Download an image from S3 and return as base64 data URI.
# #     Used by graph.py at stream time to embed the image inline.
# #     Returns None on any failure so the stream continues without the image.
# #     """
# #     try:
# #         import requests
# #         resp = requests.get(url, timeout=10)
# #         resp.raise_for_status()
# #         b64 = __import__("base64").b64encode(resp.content).decode("utf-8")
# #         return f"data:image/jpeg;base64,{b64}"
# #     except Exception as e:
# #         logger.warning(f"Failed to fetch image from {url}: {e}")
# #         return None

# """
# image_store.py — Cloud image storage abstraction for CRAG pipeline.

# Supports AWS S3 backend (IMAGE_STORE_BACKEND=s3).

# .env keys:
#     IMAGE_STORE_BACKEND=s3
#     AWS_S3_BUCKET=your-bucket-name
#     AWS_REGION=us-east-1
#     AWS_ACCESS_KEY_ID=...
#     AWS_SECRET_ACCESS_KEY=...
#     AWS_S3_PUBLIC_BASE_URL=https://your-bucket.s3.amazonaws.com

# Design notes:
#   - Images stored at full 1× quality; resizing is purely dimension-based.
#   - THUMBNAIL_MAX_SIDE=1200 and THUMBNAIL_QUALITY=88 preserve enough detail
#     for human review of ingested figures without wasting S3 storage on raw
#     300-DPI rasters.
#   - fetch_image_as_base64 uses a module-level requests.Session for connection
#     pooling, avoiding a new TCP handshake per call at stream time.
#   - boto3 client is cached per (bucket, region) so repeated uploads within
#     the same process don't re-authenticate.
# """

# import io
# import os
# import logging
# import uuid
# from functools import lru_cache
# from typing import Optional

# from PIL import Image
# from dotenv import load_dotenv

# load_dotenv()

# logger = logging.getLogger(__name__)

# BACKEND = os.getenv("IMAGE_STORE_BACKEND", "s3").lower()

# # FIX: Raised max side from 800 → 1200 and quality from 75 → 88.
# # The original 800px / q75 settings were stripping detail that the vision API
# # already paid to describe. Charts with dense axis labels or multi-series
# # legends need at least 1000px to be legible for human review via image_ref.
# # 1200px / q88 adds ~30 KB per figure (negligible S3 cost) but preserves
# # visual fidelity for downstream inspection and potential re-querying.
# THUMBNAIL_MAX_SIDE = 1200
# THUMBNAIL_QUALITY  = 88


# def _image_to_bytes(img: Image.Image) -> bytes:
#     """Resize (if oversized) and encode PIL Image as JPEG bytes."""
#     w, h = img.size
#     if max(w, h) > THUMBNAIL_MAX_SIDE:
#         scale = THUMBNAIL_MAX_SIDE / max(w, h)
#         img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
#     buf = io.BytesIO()
#     img.convert("RGB").save(buf, format="JPEG", quality=THUMBNAIL_QUALITY)
#     return buf.getvalue()


# # FIX: Cache the boto3 S3 client so it is created once per process and
# # reused across all upload calls. Previously a new client was instantiated
# # per upload, incurring repeated credential resolution and TLS handshake
# # overhead on every figure.
# @lru_cache(maxsize=4)
# def _get_s3_client(bucket: str, region: str):
#     try:
#         import boto3
#     except ImportError:
#         raise RuntimeError("boto3 not installed. Run: pip install boto3")
#     return boto3.client(
#         "s3",
#         region_name=region,
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     )


# def _upload_s3(image_bytes: bytes, key: str) -> str:
#     bucket   = os.getenv("AWS_S3_BUCKET")
#     region   = os.getenv("AWS_REGION", "us-east-1")
#     base_url = os.getenv(
#         "AWS_S3_PUBLIC_BASE_URL",
#         f"https://{bucket}.s3.{region}.amazonaws.com",
#     )

#     if not bucket:
#         raise ValueError("AWS_S3_BUCKET env var not set")

#     s3 = _get_s3_client(bucket, region)
#     s3.put_object(
#         Bucket=bucket,
#         Key=key,
#         Body=image_bytes,
#         ContentType="image/jpeg",
#         # ACL omitted — use bucket policy for public read on crag/* prefix
#     )
#     url = f"{base_url.rstrip('/')}/{key}"
#     logger.info(f"S3 upload: s3://{bucket}/{key} → {url}")
#     return url


# def upload_page_image(
#     img: Image.Image,
#     thread_id: str,
#     page_num: int,
#     fig_idx: Optional[int] = None,
# ) -> str:
#     """
#     Resize + upload a rendered image (full page or cropped figure) to S3.

#     S3 key format:
#         crag/{thread_id}/page_004_fig_2_a3f9c1b2.jpg   ← with fig_idx
#         crag/{thread_id}/page_004_a3f9c1b2.jpg          ← full-page fallback
#     """
#     image_bytes = _image_to_bytes(img)
#     unique_id   = uuid.uuid4().hex[:8]

#     if fig_idx is not None:
#         filename = f"crag/{thread_id}/page_{page_num:03d}_fig_{fig_idx}_{unique_id}.jpg"
#     else:
#         filename = f"crag/{thread_id}/page_{page_num:03d}_{unique_id}.jpg"

#     if BACKEND == "s3":
#         return _upload_s3(image_bytes, filename)
#     else:
#         raise ValueError(
#             f"Unknown IMAGE_STORE_BACKEND='{BACKEND}'. Use 's3'."
#         )


# # FIX: Module-level requests.Session for connection pooling.
# # Previously a bare requests.get() was used, which opens a new TCP connection
# # (and TLS handshake for HTTPS) for every image fetched at stream time.
# # A persistent Session reuses the underlying connection for the same host.
# _http_session: Optional[object] = None

# def _get_http_session():
#     global _http_session
#     if _http_session is None:
#         try:
#             import requests
#             _http_session = requests.Session()
#         except ImportError:
#             raise RuntimeError("requests not installed. Run: pip install requests")
#     return _http_session


# def fetch_image_as_base64(url: str) -> Optional[str]:
#     """
#     Download an image from S3 and return as a base64 data URI.
#     Used by graph.py at stream time to embed images inline.
#     Returns None on any failure so the stream continues gracefully.
#     """
#     try:
#         session = _get_http_session()
#         resp = session.get(url, timeout=10)
#         resp.raise_for_status()
#         b64 = __import__("base64").b64encode(resp.content).decode("utf-8")
#         return f"data:image/jpeg;base64,{b64}"
#     except Exception as e:
#         logger.warning(f"Failed to fetch image from {url}: {e}")
#         return None

"""
image_store.py — Cloud image storage abstraction for CRAG pipeline.

Supports AWS S3 backend (IMAGE_STORE_BACKEND=s3).

.env keys:
    IMAGE_STORE_BACKEND=s3
    AWS_S3_BUCKET=your-bucket-name
    AWS_REGION=us-east-1
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_S3_PUBLIC_BASE_URL=https://your-bucket.s3.amazonaws.com

Design notes:
  - Images stored at full 1× quality; resizing is purely dimension-based.
  - THUMBNAIL_MAX_SIDE=1200 and THUMBNAIL_QUALITY=88 preserve enough detail
    for human review of ingested figures without wasting S3 storage on raw
    300-DPI rasters.
  - fetch_image_as_base64 uses a module-level requests.Session for connection
    pooling, avoiding a new TCP handshake per call at stream time.
  - boto3 client is cached per (bucket, region) so repeated uploads within
    the same process don't re-authenticate.
"""

import io
import os
import logging
import uuid
from functools import lru_cache
from typing import Optional

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BACKEND = os.getenv("IMAGE_STORE_BACKEND", "s3").lower()

# FIX: Raised max side from 800 → 1200 and quality from 75 → 88.
# The original 800px / q75 settings were stripping detail that the vision API
# already paid to describe. Charts with dense axis labels or multi-series
# legends need at least 1000px to be legible for human review via image_ref.
# 1200px / q88 adds ~30 KB per figure (negligible S3 cost) but preserves
# visual fidelity for downstream inspection and potential re-querying.
THUMBNAIL_MAX_SIDE = 1200
THUMBNAIL_QUALITY  = 88


def _image_to_bytes(img: Image.Image) -> bytes:
    """Resize (if oversized) and encode PIL Image as JPEG bytes."""
    w, h = img.size
    if max(w, h) > THUMBNAIL_MAX_SIDE:
        scale = THUMBNAIL_MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=THUMBNAIL_QUALITY)
    return buf.getvalue()


# FIX: Cache the boto3 S3 client so it is created once per process and
# reused across all upload calls. Previously a new client was instantiated
# per upload, incurring repeated credential resolution and TLS handshake
# overhead on every figure.
@lru_cache(maxsize=4)
def _get_s3_client(bucket: str, region: str):
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed. Run: pip install boto3")
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def _upload_s3(image_bytes: bytes, key: str) -> str:
    bucket   = os.getenv("AWS_S3_BUCKET")
    region   = os.getenv("AWS_REGION", "us-east-1")
    base_url = os.getenv(
        "AWS_S3_PUBLIC_BASE_URL",
        f"https://{bucket}.s3.{region}.amazonaws.com",
    )

    if not bucket:
        raise ValueError("AWS_S3_BUCKET env var not set")

    s3 = _get_s3_client(bucket, region)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=image_bytes,
        ContentType="image/jpeg",
        # ACL omitted — use bucket policy for public read on crag/* prefix
    )
    url = f"{base_url.rstrip('/')}/{key}"
    logger.info(f"S3 upload: s3://{bucket}/{key} → {url}")
    return url


def upload_page_image(
    img: Image.Image,
    thread_id: str,
    page_num: int,
    fig_idx: Optional[int] = None,
) -> str:
    """
    Resize + upload a rendered image (full page or cropped figure) to S3.

    S3 key format:
        crag/{thread_id}/page_004_fig_2_a3f9c1b2.jpg   ← with fig_idx
        crag/{thread_id}/page_004_a3f9c1b2.jpg          ← full-page fallback
    """
    image_bytes = _image_to_bytes(img)
    unique_id   = uuid.uuid4().hex[:8]

    if fig_idx is not None:
        filename = f"crag/{thread_id}/page_{page_num:03d}_fig_{fig_idx}_{unique_id}.jpg"
    else:
        filename = f"crag/{thread_id}/page_{page_num:03d}_{unique_id}.jpg"

    if BACKEND == "s3":
        return _upload_s3(image_bytes, filename)
    else:
        raise ValueError(
            f"Unknown IMAGE_STORE_BACKEND='{BACKEND}'. Use 's3'."
        )


# FIX: Module-level requests.Session for connection pooling.
# Previously a bare requests.get() was used, which opens a new TCP connection
# (and TLS handshake for HTTPS) for every image fetched at stream time.
# A persistent Session reuses the underlying connection for the same host.
_http_session: Optional[object] = None

def _get_http_session():
    global _http_session
    if _http_session is None:
        try:
            import requests
            _http_session = requests.Session()
        except ImportError:
            raise RuntimeError("requests not installed. Run: pip install requests")
    return _http_session


def fetch_image_as_base64(url: str) -> Optional[str]:
    """
    Download an image from S3 and return as a base64 data URI.
    Used by graph.py at stream time to embed images inline.
    Returns None on any failure so the stream continues gracefully.
    """
    try:
        session = _get_http_session()
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        b64 = __import__("base64").b64encode(resp.content).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to fetch image from {url}: {e}")
        return None