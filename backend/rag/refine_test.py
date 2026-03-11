import re
import asyncio
import logging
from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ─── Sentence decomposer ───────────────────────────────────────────────────────

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ─── LLM + prompt ─────────────────────────────────────────────────────────────

class KeepOrDrop(BaseModel):
    keep: bool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


# ─── Parallel async filter ─────────────────────────────────────────────────────

MAX_CONCURRENT = 10  # Max parallel LLM calls at once — avoids rate limit spikes

async def _check_sentence(semaphore: asyncio.Semaphore, q: str, sentence: str, index: int):
    """
    Check a single sentence asynchronously under a semaphore.
    Returns (index, keep) to preserve original sentence order.
    """
    async with semaphore:
        try:
            result = await filter_chain.ainvoke({"question": q, "sentence": sentence})
            return (index, result.keep)
        except Exception as e:
            logger.warning(f"Filter call failed for sentence {index}: {e} — defaulting to keep=False")
            return (index, False)


async def _parallel_filter(q: str, strips: List[str]) -> List[str]:
    """
    Fire all sentence filter calls concurrently (up to MAX_CONCURRENT at a time).
    Returns kept sentences in original order.
    """
    if not strips:
        return []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        _check_sentence(semaphore, q, sentence, i)
        for i, sentence in enumerate(strips)
    ]

    # All tasks run in parallel, semaphore controls concurrency
    results = await asyncio.gather(*tasks)

    # Sort by original index to preserve order, then filter kept
    results.sort(key=lambda x: x[0])
    kept = [strips[i] for i, keep in results if keep]

    logger.info(f"Refine: {len(strips)} sentences → {len(kept)} kept "
                f"(dropped {len(strips) - len(kept)})")

    return kept


# ─── Sync wrapper using asyncio.run / get_event_loop ──────────────────────────

def _run_async(coro):
    """
    Run an async coroutine from sync context safely.
    Handles both cases: inside an existing event loop (FastAPI) or standalone.
    """
    try:
        loop = asyncio.get_running_loop()
        # We're inside an existing event loop (e.g. FastAPI async handler)
        # Use a thread executor to avoid "cannot run nested event loop" error
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running event loop — run directly
        return asyncio.run(coro)


# ─── Main refine node ──────────────────────────────────────────────────────────

def refine(state):
    """
    Sync refine node for LangGraph compatibility.
    Internally runs all LLM filter calls in parallel.

    Knowledge source routing:
      CORRECT   → internal docs only
      INCORRECT → web docs only
      AMBIGUOUS → internal + web docs
    """
    q = state["question"]

    verdict = state.get("verdict")
    if verdict == "CORRECT":
        docs_to_use = state.get("good_docs", [])
    elif verdict == "INCORRECT":
        docs_to_use = state.get("web_docs", [])
    else:  # AMBIGUOUS
        docs_to_use = state.get("good_docs", []) + state.get("web_docs", [])

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()

    strips = decompose_to_sentences(context)

    if not strips:
        logger.warning("Refine: no sentences to filter — returning empty context")
        return {"strips": [], "kept_strips": [], "refined_context": ""}

    # ── Parallel LLM calls ────────────────────────────────────────────────────
    kept = _run_async(_parallel_filter(q, strips))

    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }


# ─── Async refine (use this if you migrate graph nodes to async) ───────────────

async def arefine(state):
    """
    Async version of refine — drop-in replacement if your graph nodes go async.
    Avoids the ThreadPoolExecutor overhead entirely.
    """
    q = state["question"]

    verdict = state.get("verdict")
    if verdict == "CORRECT":
        docs_to_use = state.get("good_docs", [])
    elif verdict == "INCORRECT":
        docs_to_use = state.get("web_docs", [])
    else:
        docs_to_use = state.get("good_docs", []) + state.get("web_docs", [])

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    strips  = decompose_to_sentences(context)

    if not strips:
        return {"strips": [], "kept_strips": [], "refined_context": ""}

    kept = await _parallel_filter(q, strips)
    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": "\n".join(kept).strip(),
    }