import asyncio
import logging
import warnings
from typing import List, Tuple
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

UPPER_TH = 0.7
LOWER_TH = 0.3
MAX_CONCURRENT = 10  # Semaphore cap — safe for OpenAI rate limits

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class DocEvalScore(BaseModel):
    score: float
    reason: str
 
doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore, include_raw=False)


# ─── Single doc eval (async) ───────────────────────────────────────────────────
async def _eval_single(
    semaphore: asyncio.Semaphore,
    q: str,
    doc: Document,
    index: int
) -> Tuple[int, Document, float]:
    """
    Evaluate one document asynchronously under a semaphore.
    Returns (index, doc, score) to preserve original order.
    """
    async with semaphore:
        try:
            out = await doc_eval_chain.ainvoke({"question": q, "chunk": doc.page_content})
            return (index, doc, out.score)
        except Exception as e:
            logger.warning(f"Doc eval failed for index {index}: {e} — defaulting score=0.0")
            return (index, doc, 0.0)


# ─── Parallel eval ─────────────────────────────────────────────────────────────

async def _parallel_eval(q: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """
    Fire all document eval calls concurrently.
    Returns list of (doc, score) in original document order.
    """
    if not docs:
        return []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        _eval_single(semaphore, q, doc, i)
        for i, doc in enumerate(docs)
    ]

    results = await asyncio.gather(*tasks)

    # Sort by original index to preserve order
    results.sort(key=lambda x: x[0])

    logger.info(f"Evaluator: scored {len(docs)} docs in parallel")

    return [(doc, score) for _, doc, score in results]


# ─── Verdict logic (shared) ────────────────────────────────────────────────────

def _compute_verdict(scored_docs: List[Tuple[Document, float]]) -> dict:
    """
    Given (doc, score) pairs, compute good_docs + verdict.
    """
    scores = [score for _, score in scored_docs]
    good   = [doc for doc, score in scored_docs if score > LOWER_TH]

    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict":   "CORRECT",
            "reason":    f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    if scores and all(s < LOWER_TH for s in scores):
        return {
            "good_docs": [],
            "verdict":   "INCORRECT",
            "reason":    f"All retrieved chunks scored < {LOWER_TH}.",
        }

    return {
        "good_docs": good,
        "verdict":   "AMBIGUOUS",
        "reason":    f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.",
    }


# ─── Async node (use in simple_stream_graph) ───────────────────────────────────

async def aeval_each_doc_node(state) -> dict:
    """
    Async version — use with `await` inside async functions (simple_stream_graph).
    All doc evals fire in parallel.
    """
    q = state["question"]
    docs = state.get("docs", [])
    scored_docs = await _parallel_eval(q, docs)
    return _compute_verdict(scored_docs)


# ─── Sync node (use in LangGraph app_graph) ────────────────────────────────────

def eval_each_doc_node(state) -> dict:
    """
    Sync wrapper for LangGraph compatibility (app_graph nodes are sync).
    Runs parallel async evals via ThreadPoolExecutor to avoid nested loop issues.
    """
    q = state["question"]
    docs = state.get("docs", [])

    if not docs:
        return {"good_docs": [], "verdict": "INCORRECT", "reason": "No docs retrieved."}

    try:
        loop = asyncio.get_running_loop()
        # Already inside an event loop (FastAPI) — use thread executor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _parallel_eval(q, docs))
            scored_docs = future.result()
    except RuntimeError:
        # No running event loop — run directly
        scored_docs = asyncio.run(_parallel_eval(q, docs))

    return _compute_verdict(scored_docs)