# """
# speech_to_text.py — OpenAI Whisper transcription for CRAG RAG pipeline.

# Flow:
#   Frontend (hold-to-talk) → POST /threads/{thread_id}/voice
#       → transcribe audio via Whisper API
#       → resolve question against conversation history
#       → stream response via existing RAG pipeline

# Supported audio formats: webm, wav, mp3, mp4, ogg, flac, m4a
# Max file size: 25MB (Whisper API hard limit)
# """

# import io
# import os
# import logging
# from typing import Optional

# from fastapi import UploadFile, HTTPException
# from openai import AsyncOpenAI
# from pydantic import BaseModel
# from dotenv import load_dotenv 
# load_dotenv()

# logger = logging.getLogger(__name__)

# # Config 
# MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25MB — Whisper API hard limit

# # Audio MIME types browsers send via MediaRecorder
# SUPPORTED_MIME_TYPES = {
#     "audio/webm",       # Chrome / Edge default
#     "audio/ogg",        # Firefox default
#     "audio/wav",
#     "audio/wave",
#     "audio/mpeg",
#     "audio/mp3",
#     "audio/mp4",
#     "audio/x-m4a",
#     "audio/flac",
#     "video/webm",       # Some browsers tag audio-only recordings as video/webm
# }

# # OpenAI client  
# _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # ─── Response schema ──────────────────────────────────────────────────────────
# class TranscriptResult(BaseModel):
#     text: str                           # The transcribed question — feed into RAG
#     language: Optional[str] = None      # Detected language code e.g. "en", "hi"
#     duration_seconds: Optional[float] = None
#     audio_size_kb: float
#     status: str = "ok"


# # ─── Core transcription function ──────────────────────────────────────────────

# async def transcribe_audio(file: UploadFile) -> TranscriptResult:
#     """
#     Transcribe an uploaded audio file using OpenAI Whisper API.

#     Args:
#         file: FastAPI UploadFile — audio blob from the frontend hold-to-talk button.

#     Returns:
#         TranscriptResult with .text ready to pass into simple_stream_graph as the question.

#     Raises:
#         HTTPException 400 for invalid/empty/oversized audio.
#         HTTPException 500 for Whisper API failures.
#     """

#     # ── 1. Read and validate ──────────────────────────────────────────────────
#     contents = await file.read()

#     if not contents:
#         raise HTTPException(status_code=400, detail="Audio file is empty.")

#     if len(contents) > MAX_AUDIO_BYTES:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Audio too large ({len(contents) / (1024*1024):.1f}MB). Max is 25MB."
#         )

#     content_type = file.content_type or "audio/webm"
#     if content_type not in SUPPORTED_MIME_TYPES:
#         # Log a warning but don't hard-reject — browsers are inconsistent with MIME types
#         logger.warning(
#             f"Unexpected audio content_type '{content_type}' — attempting transcription anyway."
#         )

#     size_kb = len(contents) / 1024
#     filename = file.filename or _infer_filename(content_type)

#     logger.info(f"🎙️ Transcribing audio: {filename} ({size_kb:.1f}KB, type={content_type})")

#     # ── 2. Call Whisper API ───────────────────────────────────────────────────
#     try:
#         # Whisper API requires a named file-like object so it can detect the format
#         audio_file = (filename, io.BytesIO(contents), content_type)

#         response = await _client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file,
#             response_format="verbose_json",  # gives language + duration on top of text
#             # language="en"
#         )

#     except Exception as e:
#         logger.exception("Whisper API call failed")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Transcription failed: {str(e)}"
#         )

#     # ── 3. Extract and validate transcript ───────────────────────────────────
#     text = (response.text or "").strip()

#     if not text:
#         raise HTTPException(
#             status_code=422,
#             detail="Whisper returned an empty transcript. Was there any speech in the recording?"
#         )

#     language = getattr(response, "language", None)
#     duration = getattr(response, "duration", None)

#     logger.info(
#         f"✅ Transcribed ({language or 'unknown'}, {duration:.1f}s): \"{text[:80]}{'...' if len(text) > 80 else ''}\""
#     )

#     return TranscriptResult(
#         text=text,
#         language=language,
#         duration_seconds=duration,
#         audio_size_kb=round(size_kb, 1),
#     )


# # ─── Helper ───────────────────────────────────────────────────────────────────

# def _infer_filename(content_type: str) -> str:
#     """Map MIME type → sensible filename for the Whisper API."""
#     mapping = {
#         "audio/webm":  "audio.webm",
#         "video/webm":  "audio.webm",
#         "audio/ogg":   "audio.ogg",
#         "audio/wav":   "audio.wav",
#         "audio/wave":  "audio.wav",
#         "audio/mpeg":  "audio.mp3",
#         "audio/mp3":   "audio.mp3",
#         "audio/mp4":   "audio.mp4",
#         "audio/x-m4a": "audio.m4a",
#         "audio/flac":  "audio.flac",
#     }
#     return mapping.get(content_type, "audio.webm")


"""
speech_to_text.py — Local Whisper tiny transcription for CRAG RAG pipeline.

Flow:
  Frontend (hold-to-talk) → POST /threads/{thread_id}/voice
      → transcribe audio via local Whisper tiny model
      → resolve question against conversation history
      → stream response via existing RAG pipeline

Supported audio formats: webm, wav, mp3, mp4, ogg, flac, m4a
Max file size: 25MB (Whisper API hard limit, kept for consistency)
"""

import io
import os
import logging
import tempfile
import asyncio
from typing import Optional

import whisper
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25MB — Whisper API hard limit

# Audio MIME types browsers send via MediaRecorder
SUPPORTED_MIME_TYPES = {
    "audio/webm",       # Chrome / Edge default
    "audio/ogg",        # Firefox default
    "audio/wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/flac",
    "video/webm",       # Some browsers tag audio-only recordings as video/webm
}

# ----------------------------------------------------------------------
# Load Whisper model once at startup (tiny variant)
# ----------------------------------------------------------------------
_model = None

def get_whisper_model():
    """Lazy-load the Whisper model to avoid slow import at module level."""
    global _model
    if _model is None:
        logger.info("Loading Whisper tiny model...")
        _model = whisper.load_model("small")
        logger.info("Whisper tiny model loaded.")
    return _model

# ----------------------------------------------------------------------
# Response schema
# ----------------------------------------------------------------------
class TranscriptResult(BaseModel):
    text: str                           # The transcribed question – feed into RAG
    language: Optional[str] = None      # Detected language code e.g. "en", "hi"
    duration_seconds: Optional[float] = None
    audio_size_kb: float
    status: str = "ok"

# ----------------------------------------------------------------------
# Core transcription function
# ----------------------------------------------------------------------
async def transcribe_audio(file: UploadFile) -> TranscriptResult:
    """
    Transcribe an uploaded audio file using a local Whisper tiny model.

    Args:
        file: FastAPI UploadFile — audio blob from the frontend hold-to-talk button.

    Returns:
        TranscriptResult with .text ready to pass into simple_stream_graph as the question.

    Raises:
        HTTPException 400 for invalid/empty/oversized audio.
        HTTPException 500 for Whisper failures.
    """
    # ── 1. Read and validate ──────────────────────────────────────────────
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    if len(contents) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too large ({len(contents) / (1024*1024):.1f}MB). Max is 25MB."
        )

    content_type = file.content_type or "audio/webm"
    if content_type not in SUPPORTED_MIME_TYPES:
        # Log a warning but don't hard-reject — browsers are inconsistent with MIME types
        logger.warning(
            f"Unexpected audio content_type '{content_type}' — attempting transcription anyway."
        )

    size_kb = len(contents) / 1024
    filename = file.filename or _infer_filename(content_type)

    logger.info(f"🎙️ Transcribing audio: {filename} ({size_kb:.1f}KB, type={content_type})")

    # ── 2. Save to a temporary file for Whisper ───────────────────────────
    # Whisper's load_audio() expects a file path, not bytes in memory.
    # Using a temporary file ensures cleanup after we're done.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        logger.exception("Failed to write temporary audio file")
        raise HTTPException(status_code=500, detail="Could not save audio for processing.")

    # ── 3. Run transcription in a thread pool ─────────────────────────────
    model = get_whisper_model()
    try:
        # asyncio.to_thread runs the synchronous whisper.transcribe in a separate thread
        result = await asyncio.to_thread(
            model.transcribe,
            tmp_path,
            language=None,          # let Whisper auto-detect
            task="transcribe",
            fp16=False              # use FP32 for CPU compatibility; set True if GPU
        )
    except Exception as e:
        logger.exception("Whisper transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass  # ignore cleanup errors

    # ── 4. Extract and validate transcript ────────────────────────────────
    text = (result.get("text") or "").strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="Whisper returned an empty transcript. Was there any speech in the recording?"
        )

    language = result.get("language")
    # Duration can be approximated from the audio file or from the sum of segment durations.
    # We'll compute it from the segments if available, otherwise set to None.
    duration = None
    if "segments" in result and result["segments"]:
        # duration is roughly the end time of the last segment
        duration = result["segments"][-1].get("end", None)

    logger.info(
        f"✅ Transcribed ({language or 'unknown'}, {duration:.1f}s): "
        f"\"{text[:80]}{'...' if len(text) > 80 else ''}\""
    )

    return TranscriptResult(
        text=text,
        language=language,
        duration_seconds=duration,
        audio_size_kb=round(size_kb, 1),
    )

# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def _infer_filename(content_type: str) -> str:
    """Map MIME type → sensible filename for Whisper (file extension)."""
    mapping = {
        "audio/webm":  "audio.webm",
        "video/webm":  "audio.webm",
        "audio/ogg":   "audio.ogg",
        "audio/wav":   "audio.wav",
        "audio/wave":  "audio.wav",
        "audio/mpeg":  "audio.mp3",
        "audio/mp3":   "audio.mp3",
        "audio/mp4":   "audio.mp4",
        "audio/x-m4a": "audio.m4a",
        "audio/flac":  "audio.flac",
    }
    return mapping.get(content_type, "audio.webm") 