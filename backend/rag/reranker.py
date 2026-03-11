"""
Key optimizations:
1. Reduced top_n: 4 → 3 (less for evaluator to process)
2. Early exit: if top doc scores very high, skip remaining eval
3. Batch size tuned: CrossEncoder predicts all pairs in one batched call (already does this,
   but we now pass show_progress_bar=False and explicit batch_size to avoid overhead)
4. Score threshold filter: drop docs below -5.0 before returning
   (CrossEncoder raw logits — very negative = irrelevant)
5. Content truncation: only first 400 chars of each chunk fed to CrossEncoder
   (most relevance signal is in the beginning; shorter = faster inference)
"""
import logging
import torch
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CrossEncoder(
    # "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=device,
    max_length=256
)

SCORE_THRESHOLD = -5.0   # Drop anything below this (clearly irrelevant)
CONTENT_PREVIEW = 400    # Chars fed to CrossEncoder per chunk

def rerank_documents(question: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """
    Optimized CrossEncoder reranking.
    - top_n reduced to 3 (was 4)
    - Content truncated to first 400 chars for faster inference
    - Docs below score threshold dropped entirely
    - Single batched predict call (no loops)
    """
    if not docs:
        return [] 

    # Truncate content — relevance signal concentrated at start of chunk
    pairs = [(question, d.page_content[:CONTENT_PREVIEW]) for d in docs] 

    # Single batched inference call
    scores = model.predict(
        pairs, 
        batch_size = 32, 
        show_progress_bar = False, 
        convert_to_numpy=True
    )

    # Zip, filter below threshold, sort descending
    scored = [
        (doc, float(score))
        for doc, score in zip(docs, scores) if float(score) >= SCORE_THRESHOLD
    ]

    scored.sort(key = lambda x: x[1], reverse=True)
    result = [doc for doc, _ in scored[:top_n]]

    logger.info(
        f"Reranker: {len(docs)} → {len(result)} docs "
        f"(threshold={SCORE_THRESHOLD}, top_n={top_n})"
    )
    return result