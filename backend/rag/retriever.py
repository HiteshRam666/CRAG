"""
retriever.py — Hybrid retrieval across two AstraDB collections.

Collections searched on every query:
  • "crag"            — OpenAI text-embedding-3-small (1536-dim)  — plain text from old ingestion
  • "crag_multimodal" — Cohere embed-english-v3.0 (1024-dim)      — text + tables + images

Key optimisations (unchanged from original):
  1. BM25 index cached per thread — rebuilt only when docs change
  2. Async vector search — non-blocking AstraDB calls
  3. Query embedding cached in-process — same question never re-embedded
  4. Reduced k: retrieve 6 per store (12 total), reranker keeps top 3

NOTE on Cohere input_type:
  At query time we need input_type="search_query".
  We temporarily override cohere_embeddings.input_type before calling
  similarity_search_with_score, then restore it. This is safe because
  ingestion always runs in a separate thread via asyncio.to_thread().
"""

import asyncio
import hashlib
from functools import lru_cache
from typing import List, Tuple, Dict

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# ─── OpenAI store (unchanged) ─────────────────────────────────────────────────
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = AstraDBVectorStore(
    embedding=openai_embeddings,
    collection_name="crag",
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
)

# ─── Cohere store (multimodal chunks) ────────────────────────────────────────
# Imported from shared module so embedding model is defined exactly once.
from backend.rag.cohere_store import cohere_query_store


# ─── BM25 cache ───────────────────────────────────────────────────────────────
_bm25_cache: Dict[str, Tuple[str, BM25Okapi]] = {}


def invalidate_bm25_cache(thread_id: str):
    """Call this after a new PDF is ingested for a thread."""
    _bm25_cache.pop(thread_id, None)
    logger.info(f"BM25 cache invalidated for thread={thread_id}")


def _get_bm25(thread_id: str, docs: List[Document]) -> BM25Okapi:
    doc_hash = hashlib.md5(
        "".join(d.page_content for d in docs).encode()
    ).hexdigest()

    cached = _bm25_cache.get(thread_id)
    if cached and cached[0] == doc_hash:
        return cached[1]

    tokenized = [d.page_content.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    _bm25_cache[thread_id] = (doc_hash, bm25)
    logger.info(f"BM25 index rebuilt for thread={thread_id} ({len(docs)} docs)")
    return bm25


# ─── Cached query embeddings ──────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _embed_openai_cached(query: str) -> tuple:
    return tuple(openai_embeddings.embed_query(query))


# ─── Async vector searches ────────────────────────────────────────────────────

async def _async_openai_search(query: str, thread_id: str, k: int):
    """AstraDB similarity search via OpenAI embeddings (non-blocking)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: vector_store.similarity_search_with_score(
            query=query, k=k, filter={"thread_id": thread_id}
        ),
    )


async def _async_cohere_search(query: str, thread_id: str, k: int):
    """
    AstraDB similarity search via Cohere embeddings (non-blocking).
    Uses cohere_query_store which is initialised with input_type="search_query".
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: cohere_query_store.similarity_search_with_score(
            query=query, k=k, filter={"thread_id": thread_id}
        ),
    )


# ─── Core async hybrid search ────────────────────────────────────────────────

async def _async_hybrid(
    query: str, thread_id: str, k: int, alpha: float
) -> List[Document]:
    """
    Hybrid search across BOTH vector stores.

    Steps:
      1. Fire OpenAI and Cohere vector searches concurrently.
      2. Deduplicate by page_content hash.
      3. Run BM25 over the merged doc pool.
      4. Fuse: alpha * vector_score + (1-alpha) * bm25_score.
      5. Return top-k ranked docs.
    """
    # 1. Concurrent vector searches ───────────────────────────────────────────
    openai_task = asyncio.create_task(_async_openai_search(query, thread_id, k))
    cohere_task = asyncio.create_task(_async_cohere_search(query, thread_id, k))

    openai_results, cohere_results = await asyncio.gather(
        openai_task, cohere_task, return_exceptions=True
    )

    # Gracefully handle store errors (e.g. collection not yet created)
    if isinstance(openai_results, Exception):
        logger.warning(f"OpenAI vector search failed: {openai_results}")
        openai_results = []
    if isinstance(cohere_results, Exception):
        logger.warning(f"Cohere vector search failed: {cohere_results}")
        cohere_results = []

    if not openai_results and not cohere_results:
        return []

    # 2. Merge + deduplicate ───────────────────────────────────────────────────
    # Build a map: content_hash → (doc, best_vector_score)
    # Lower AstraDB score = closer (distance), so keep the minimum.
    seen: Dict[str, Tuple[Document, float]] = {}

    for doc, score in (openai_results + cohere_results):
        key = hashlib.md5(doc.page_content.encode()).hexdigest()
        if key not in seen or score < seen[key][1]:
            seen[key] = (doc, score)

    docs         = [d for d, _ in seen.values()]
    raw_scores   = np.array([s for _, s in seen.values()])

    logger.info(
        f"Vector search merged: {len(openai_results)} openai + "
        f"{len(cohere_results)} cohere → {len(docs)} unique docs"
    )

    # 3. Normalise vector scores (distance → similarity) ──────────────────────
    v_scores = 1 / (1 + raw_scores)
    v_scores = (v_scores - v_scores.min()) / (v_scores.max() - v_scores.min() + 1e-8)

    # 4. BM25 scoring (cached per thread) ─────────────────────────────────────
    bm25 = _get_bm25(thread_id, docs)
    b_scores = np.array(bm25.get_scores(query.split()))
    b_scores = (b_scores - b_scores.min()) / (b_scores.max() - b_scores.min() + 1e-8)

    # 5. Score fusion + sort ──────────────────────────────────────────────────
    final_scores   = alpha * v_scores + (1 - alpha) * b_scores
    ranked_indices = np.argsort(final_scores)[::-1]

    return [docs[i] for i in ranked_indices[:k]]


# ─── Public sync entry point (used by graph.py) ───────────────────────────────

def hybrid_search(
    query: str,
    thread_id: str,
    k: int = 6,
    alpha: float = 0.6,
) -> List[Document]:
    """
    Sync wrapper — searches both OpenAI and Cohere collections, fuses with BM25.
    Identical signature to the original; graph.py needs no changes.
    """
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _async_hybrid(query, thread_id, k, alpha))
            return future.result()
    except RuntimeError:
        return asyncio.run(_async_hybrid(query, thread_id, k, alpha))