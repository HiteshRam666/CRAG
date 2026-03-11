"""
main.py - FastAPI with image output support for visual PDF chunks
"""
import uuid
import asyncio
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from backend.rag.speech_to_text import TranscriptResult, transcribe_audio
from backend.rag.mongo_cache import (
    create_thread,
    thread_exists,
    get_recent_threads,
    get_thread_metadata,
    delete_thread,
    save_feedback,
    cache_invalidate_thread
)
from backend.rag.graph import simple_stream_graph, get_graph_stats
from backend.ingest.pdf_ingest import ingest_pdf
import os
import uvicorn
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CRAG Chatbot", version="4.1.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models
# ============================================================================

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

class ThreadCreateResponse(BaseModel):
    thread_id: str
    title: str
    created_at: str

class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    last_active: str
    message_count: int
    topics: List[str]
    preview: Optional[str] = None

class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = None

class GraphStatsResponse(BaseModel):
    graph_type: str
    nodes: List[str]
    checkpointing: bool
    streaming: bool
    compiled: bool

class VoiceChatResponse(BaseModel):
    thread_id: str
    transcript: str
    language: Optional[str]
    duration_seconds: Optional[float]
    audio_size_kb: float
    answer: str


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 CRAG Chatbot v4.1 starting...")
    if os.getenv("LANGSMITH_API_KEY"):
        logger.info(f"✅ LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
    else:
        logger.warning("⚠️ LangSmith not configured - check your .env file")
    stats = await get_graph_stats()
    logger.info(f"📊 Graph stats: {stats}")
    logger.info("✅ Ready to accept requests!")

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "4.1.0",
        "features": ["pdf", "image_output"],
        "database": "MongoDB Atlas",
        "graph": "hybrid (custom + LangGraph)",
        "streaming": "ready"
    }

@app.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats():
    return await get_graph_stats()

# ============================================================================
# Thread Management
# ============================================================================

@app.post("/threads", response_model=ThreadCreateResponse)
async def create_new_thread(title: Optional[str] = None):
    thread_id = str(uuid.uuid4())
    if not title:
        title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    create_thread(thread_id, title)
    logger.info(f"Thread created: {thread_id}")
    return ThreadCreateResponse(
        thread_id=thread_id,
        title=title,
        created_at=datetime.now().isoformat()
    )

@app.get("/threads", response_model=List[ThreadInfo])
async def list_threads(limit: int = 20):
    return get_recent_threads(limit)

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    if not thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    from backend.rag.mongo_cache import load_memory
    messages = await load_memory(thread_id, limit=100)
    metadata = get_thread_metadata(thread_id)
    return {"thread_id": thread_id, "metadata": metadata, "messages": messages}

@app.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    if not thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    delete_thread(thread_id)
    return {"status": "deleted", "thread_id": thread_id}

# ============================================================================
# PDF Upload
# ============================================================================

@app.post("/threads/{thread_id}/upload")
async def upload_pdf(thread_id: str, file: UploadFile = File(...)):
    """Upload and multimodal-index a PDF for this thread."""
    if not thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")
    try:
        await asyncio.to_thread(ingest_pdf, contents, thread_id)
        cache_invalidate_thread(thread_id)
        logger.info(f"PDF uploaded and indexed for thread {thread_id}")
        return {"status": "PDF indexed successfully", "thread_id": thread_id}
    except Exception as e:
        logger.exception("PDF ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Streaming Chat — with inline image output
# ============================================================================

async def stream_generator(question: str, thread_id: str):
    """
    Streams the RAG answer token by token.
    If the answer is sourced from a visual PDF chunk, the rendered page image
    is appended inline as a base64 data URI after the text answer.

    The image renders in any Markdown-capable frontend using:
        ![Page N](data:image/jpeg;base64,...)
    """
    try:
        yield f"📝 **Thread:** {thread_id[:8]}...\n"
        yield "═" * 50 + "\n\n"

        image_ref = None

        async for token in simple_stream_graph({
            "question":  question,
            "thread_id": thread_id,
        }):
            # graph.py yields str tokens for text, and a dict signal for images
            if isinstance(token, dict) and token.get("type") == "image_ref":
                image_ref = token["data"]   # base64 data URI fetched from S3
            else:
                yield token

        # Append image inline after the full text answer
        if image_ref:
            yield "\n\n" + "─" * 40 + "\n"
            yield "🖼️ **Relevant page image:**\n"
            yield f"![Page image]({image_ref})\n"

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield f"\n\n❌ Error: {str(e)}\n"


@app.post("/threads/{thread_id}/chat/stream")
async def chat_stream(thread_id: str, req: ChatRequest):
    """
    Main streaming chat endpoint.
    If the answer cites a visual PDF chunk (chart, diagram, image),
    the rendered page is appended inline as a base64 image after the text.

    Usage:
    curl -N -X POST "http://localhost:8000/threads/YOUR_THREAD_ID/chat/stream" \\
         -H "Content-Type: application/json" \\
         -d '{"question":"Show me the revenue chart"}'
    """
    if not thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    logger.info(f"📨 Streaming request for thread {thread_id}: {req.question[:100]}...")
    return StreamingResponse(
        stream_generator(req.question, thread_id),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-ID": thread_id,
            "X-Graph-Type": "hybrid-langgraph",
            "Access-Control-Allow-Origin": "*",
        }
    )

# ============================================================================
# Feedback
# ============================================================================

@app.post("/threads/{thread_id}/messages/{message_id}/feedback")
async def submit_feedback(thread_id: str, message_id: str, req: FeedbackRequest):
    if not thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    await save_feedback(thread_id, message_id, req.rating, req.feedback)
    return {"status": "feedback saved"}

# ============================================================================
# Cache Management
# ============================================================================

@app.delete("/threads/{thread_id}/cache")
async def clear_thread_cache(thread_id: str):
    if not thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    cache_invalidate_thread(thread_id)
    return {"status": "cache cleared", "thread_id": thread_id}

# ============================================================================
# Voice Chat
# ============================================================================

@app.post("/threads/{thread_id}/voice/stream")
async def voice_chat_stream(thread_id: str, file: UploadFile = File(...)):
    if not thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    result: TranscriptResult = await transcribe_audio(file)
    logger.info(f"🎙️ Voice — thread={thread_id[:8]} | transcript='{result.text[:80]}'")

    async def _voice_stream():
        yield f"🎙️ *Heard:* \"{result.text}\"\n"
        yield "═" * 50 + "\n\n"
        async for token in simple_stream_graph({
            "question":  result.text,
            "thread_id": thread_id,
        }):
            # Skip image signals for voice stream (audio-only interface)
            if not (isinstance(token, dict) and token.get("type") == "image_ref"):
                yield token

    return StreamingResponse(
        _voice_stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-ID": thread_id,
            "X-Transcript": result.text[:500],
            "X-Language": result.language or "unknown",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "X-Transcript, X-Language, X-Thread-ID",
        }
    )


@app.post("/threads/{thread_id}/voice", response_model=VoiceChatResponse)
async def voice_chat(thread_id: str, file: UploadFile = File(...)):
    if not thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    result: TranscriptResult = await transcribe_audio(file)
    full_answer = ""
    async for token in simple_stream_graph({
        "question":  result.text,
        "thread_id": thread_id,
    }):
        if not (isinstance(token, dict) and token.get("type") == "image_ref"):
            full_answer += token
    return VoiceChatResponse(
        thread_id=thread_id,
        transcript=result.text,
        language=result.language,
        duration_seconds=result.duration_seconds,
        audio_size_kb=result.audio_size_kb,
        answer=full_answer,
    )


@app.post("/transcribe")
async def transcribe_only(file: UploadFile = File(...)) -> TranscriptResult:
    return await transcribe_audio(file)