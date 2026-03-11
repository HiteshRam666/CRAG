"""
graph.py - Hybrid approach with mixed PDF and Web source support
"""

from typing import TypedDict, List, Optional, AsyncGenerator, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
import asyncio
import logging
import time

from backend.rag.retriever import hybrid_search
from backend.rag.refine_test import arefine
from backend.rag.evaluator_test import aeval_each_doc_node
from backend.rag.generator import generate_stream
from backend.rag.web import web_search_node, rewrite_query
from backend.rag.reranker import rerank_documents
from backend.rag.context_resolver import resolve_question
from backend.ingest.image_store import fetch_image_as_base64
from backend.rag.mongo_cache import (
    get_conversation_context,
    save_message,
    cache_get,
    cache_set,
    load_memory
)
logger = logging.getLogger(__name__)

# State Definition
class GraphState(TypedDict, total=False):
    """Internal state for LangGraph"""
    question: str
    original_question: str
    thread_id: str
    docs: List[Document]
    verdict: str
    good_docs: List[Document]
    web_query: str
    web_docs: List[Document]
    refined_context: str
    source_type: str  # "pdf", "web", "mixed", "cache", "none"
    source_urls: List[str]  # For web sources
    source_titles: List[str]  # Titles for web sources
    pdf_used: bool  # Track if PDFs were ACTUALLY USED in final answer
    web_used: bool  # Track if web sources were ACTUALLY USED in final answer
    image_refs: List[str]  # base64 data URIs for visual chunks used in answer
    error: Optional[str]
    start_time: float
    processing_stages: Dict[str, float]

# Node Functions with Performance Tracking
async def retrieve_node(state: GraphState) -> Dict:
    """Async retrieve node with timing"""
    start = time.time()
    thread_id = state.get("thread_id")
    question = state["question"]
    
    logger.debug(f"Retrieve node starting for: {question[:50]}...")
    
    # Run hybrid search and rerank in thread pool
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(
        None,
        lambda: hybrid_search(query=question, thread_id=thread_id, k=6, alpha=0.6)
    )
    reranked = await loop.run_in_executor(
        None,
        lambda: rerank_documents(question, docs, top_n=3)
    )
    
    elapsed = time.time() - start
    
    # Don't set pdf_used=True here - we'll set it ONLY if verdict is CORRECT
    # This prevents false positives where PDFs are retrieved but not used
    source_type = "pdf" if reranked else "none"
    
    logger.info(f"✅ Retrieve node completed in {elapsed:.2f}s - found {len(reranked)} PDF docs")
    
    return {
        "docs": reranked,
        "source_type": source_type,
        "pdf_used": False,  # Initially False - will be set to True only if verdict is CORRECT
        "processing_stages": {"retrieve": elapsed}
    }

async def evaluate_node(state: GraphState) -> Dict:
    """Evaluate retrieved documents"""
    start = time.time()
    logger.debug(f"Evaluate node starting with {len(state.get('docs', []))} docs")
    
    result = await aeval_each_doc_node(state)
    
    elapsed = time.time() - start
    logger.info(f"✅ Evaluate node completed in {elapsed:.2f}s - verdict: {result.get('verdict')}")
    
    # If verdict is CORRECT, mark PDF as actually used
    if result.get("verdict") == "CORRECT":
        result["pdf_used"] = True
        logger.info("📄 PDF documents will be used in answer (verdict: CORRECT)")
    
    # Add timing
    result["processing_stages"] = {"evaluate": elapsed}
    return result

# def route_node(state: GraphState) -> str:
#     """Route based on verdict and track source types"""
#     verdict = state.get("verdict", "AMBIGUOUS")
#     current_source = state.get("source_type", "none")
#     pdf_used = state.get("pdf_used", False)  # This will be True only if verdict was CORRECT
    
#     logger.debug(f"Routing based on verdict: {verdict}, current source: {current_source}, pdf_used: {pdf_used}")
    
#     # Decision logic for mixed sources
#     if verdict == "CORRECT" and current_source == "pdf":
#         # We have good PDF docs that will be used
#         return "refine"
    
#     # If verdict is AMBIGUOUS or INCORRECT, we need web search
#     # Even if we have PDFs (but they're not sufficient), we'll still search web
#     return "rewrite"

# Add to route_node function to track follow-up depth
def route_node(state: GraphState) -> str:
    """Route based on verdict and track source types"""
    verdict = state.get("verdict", "AMBIGUOUS")
    current_source = state.get("source_type", "none")
    pdf_used = state.get("pdf_used", False)
    
    # Track follow-up depth in conversation context
    context = state.get("conversation_context", {})
    current_state = context.get("current_state", {})
    follow_up_count = current_state.get("follow_up_count", 0)
    
    # Update follow-up count based on question type
    # (This would be set by the resolver if it's a follow-up)
    
    logger.debug(f"Routing based on verdict: {verdict}, current source: {current_source}, pdf_used: {pdf_used}")
    
    # Decision logic for mixed sources
    if verdict == "CORRECT" and current_source == "pdf":
        return "refine"
    
    return "rewrite"

async def rewrite_node(state: GraphState) -> Dict:
    """Rewrite query for web search"""
    start = time.time()
    logger.debug(f"Rewrite node starting")
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: rewrite_query(state))
    
    elapsed = time.time() - start
    logger.info(f"✅ Rewrite node completed in {elapsed:.2f}s - query: {result.get('web_query', '')}")
    
    result["processing_stages"] = {"rewrite": elapsed}
    return result

async def web_search_node_async(state: GraphState) -> Dict:
    """Perform web search and track web sources"""
    start = time.time()
    logger.debug(f"Web search node starting")
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: web_search_node(state))
    
    # Extract URLs and titles
    urls = []
    titles = []
    if result.get("web_docs"):
        for doc in result["web_docs"]:
            if doc.metadata and doc.metadata.get("url"):
                urls.append(doc.metadata["url"])
                titles.append(doc.metadata.get("title", "Web Source"))
        
        logger.info(f"Found {len(urls)} web sources")
    
    elapsed = time.time() - start
    
    # Check if web docs were actually found
    web_used = bool(urls)
    
    # Get PDF usage from state - this will be True ONLY if verdict was CORRECT
    pdf_used = state.get("pdf_used", False)
    
    # Determine final source type based on ACTUAL usage
    if pdf_used and web_used:
        source_type = "mixed"
        logger.info("📚 Mixed sources will be used (PDF + Web)")
    elif pdf_used:
        source_type = "pdf"
        logger.info("📄 PDF-only sources will be used")
    elif web_used:
        source_type = "web"
        logger.info("🌐 Web-only sources will be used")
    else:
        source_type = "none"
        logger.info("⚠️ No sources found")
    
    # Return with source information
    return {
        "web_docs": result.get("web_docs", []),
        "source_type": source_type,
        "source_urls": urls,
        "source_titles": titles,
        "web_used": web_used,
        "pdf_used": pdf_used,  # Preserve PDF usage flag (only True if verdict was CORRECT)
        "processing_stages": {"web_search": elapsed}
    }

async def refine_node(state: GraphState) -> Dict:
    """Refine context"""
    start = time.time()
    logger.debug(f"Refine node starting")
    
    result = await arefine(state)
    
    # Collect image_refs from any image-type docs that made it into the answer
    verdict = state.get("verdict", "AMBIGUOUS")
    if verdict == "CORRECT":
        docs_used = state.get("good_docs", [])
    elif verdict == "INCORRECT":
        docs_used = state.get("web_docs", [])
    else:
        docs_used = state.get("good_docs", []) + state.get("web_docs", [])

    image_refs = [
        d.metadata["image_ref"]
        for d in docs_used
        if d.metadata.get("content_type") == "image" and d.metadata.get("image_ref")
    ]
    result["image_refs"] = image_refs

    elapsed = time.time() - start
    logger.info(f"✅ Refine node completed in {elapsed:.2f}s — {len(image_refs)} image refs collected")
    
    result["processing_stages"] = {"refine": elapsed}
    return result

# Build the LangGraph (Internal)

# Create graph builder
builder = StateGraph(GraphState)

# Add nodes
builder.add_node("retrieve", retrieve_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("web_search", web_search_node_async)
builder.add_node("refine", refine_node)

# Add edges
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "evaluate")

# Conditional routing
builder.add_conditional_edges(
    "evaluate",
    route_node,
    {
        "refine": "refine",
        "rewrite": "rewrite",
    }
)

builder.add_edge("rewrite", "web_search")
builder.add_edge("web_search", "refine")
builder.add_edge("refine", END)

# Add memory saver for checkpointing (optional)
memory = MemorySaver()

# Compile the graph (do this once at startup)
_internal_graph = builder.compile(checkpointer=memory)
logger.info("✅ LangGraph compiled successfully")

# ============================================================================
# YOUR FAMILIAR simple_stream_graph FUNCTION
# (Keeps the same interface you know and love!)
# ============================================================================
async def simple_stream_graph(inputs: dict) -> AsyncGenerator[str, None]:
    """
    YOUR ORIGINAL INTERFACE - unchanged!
    But now powered by LangGraph under the hood with mixed source support.
    
    Args:
        inputs: Dictionary with 'question' and 'thread_id'
        
    Yields:
        Tokens as they're generated
    """
    # Extract inputs
    thread_id = inputs.get("thread_id")
    original_question = inputs["question"]
    
    # Track overall performance
    overall_start = time.time()
    stage_times = {}
    
    try:
        # ====================================================================
        # STEP 1: Get rich conversation context from MongoDB
        # ====================================================================
        stage_start = time.time()
        context = await get_conversation_context(thread_id) if thread_id else {} 
        stage_times["get_context"] = time.time() - stage_start
        logger.debug(f"Context loaded in {stage_times['get_context']:.2f}s")

        # ====================================================================
        # STEP 2: Resolve question with context (your existing resolver)
        # ====================================================================
        stage_start = time.time()
        recent_messages = context.get('recent_messages', []) if context else [] 
        resolved = await resolve_question(
            original_question, 
            recent_messages, 
            context.get("user_profile", {})
        )
        stage_times["resolve"] = time.time() - stage_start
        
        if resolved != original_question:
            logger.info(f"🔄 Resolved: '{original_question}' → '{resolved}'")
        else:
            logger.debug(f"Question unchanged: '{original_question}'")

        # ====================================================================
        # STEP 3: Check MongoDB cache (fast path)
        # ====================================================================
        stage_start = time.time()
        
        if thread_id:
            cached = await cache_get(thread_id, resolved)
            if cached:
                answer, from_semantic = cached
                cache_type = "semantic" if from_semantic else "exact"
                stage_times["cache_check"] = time.time() - stage_start
                
                logger.info(f"⚡ Cache HIT ({cache_type}) in {stage_times['cache_check']:.2f}s")
                
                # Return cached answer
                yield answer
                
                # Add cache footer
                yield "\n\n" + "─" * 40 + "\n"
                yield f"⚡ *(answered from {cache_type} cache)*"
                
                # Save to conversation history (background task)
                asyncio.create_task(save_message(
                    thread_id=thread_id, 
                    role="user", 
                    content=original_question, 
                    tokens=len(original_question.split())
                ))
                asyncio.create_task(save_message(
                    thread_id=thread_id,
                    role="assistant",
                    content=answer,
                    tokens=len(answer.split()),
                    metadata={"from_cache": True, "cache_type": cache_type}
                ))
                
                # Log total time
                logger.info(f"✅ Total request time (cache hit): {time.time() - overall_start:.2f}s")
                return
            else:
                stage_times["cache_check"] = time.time() - stage_start
                logger.info("⚪ Cache MISS")
        else:
            stage_times["cache_check"] = time.time() - stage_start

        # ====================================================================
        # STEP 4: Prepare state for LangGraph
        # ====================================================================
        initial_state = {
            "question": resolved,
            "original_question": original_question,
            "thread_id": thread_id,
            "source_type": "none",
            "source_urls": [],
            "source_titles": [],
            "pdf_used": False,
            "web_used": False,
            "image_refs": [],
            "start_time": time.time(),
            "processing_stages": {}
        }

        # ====================================================================
        # STEP 5: Run LangGraph internally
        # ====================================================================
        logger.info("🚀 Starting LangGraph pipeline...")
        graph_start = time.time()
        
        # Create a config with thread_id for checkpointing
        # config = {"configurable": {"thread_id": thread_id}}
        
        # Store results from graph
        graph_results = {}
        source_type = "none"
        source_urls = []
        source_titles = []
        pdf_used = False
        web_used = False
        image_refs_collected = []

        config = {
            "configurable": {
                "thread_id": thread_id,
                "run_name": f"crag_thread_{thread_id[:8]}",  # Adds readable run names
                "tags": ["crag", "production", "hybrid"]     # Adds searchable tags
            }
        }
        
        # Run the graph and collect results
        async for event in _internal_graph.astream(
            initial_state,
            config,
            stream_mode="values"
        ):
            # Update our state with graph results
            if isinstance(event, dict):
                graph_results.update(event)
                
                # Capture source information as they become available
                if event.get("source_type"):
                    source_type = event["source_type"]
                if event.get("source_urls"):
                    source_urls = event["source_urls"]
                if event.get("source_titles"):
                    source_titles = event["source_titles"]
                if event.get("pdf_used") is not None:
                    pdf_used = event["pdf_used"]  # This will be True ONLY if verdict was CORRECT
                if event.get("web_used") is not None:
                    web_used = event["web_used"]
                if event.get("image_refs"):
                    image_refs_collected = event["image_refs"]
        
        graph_time = time.time() - graph_start
        logger.info(f"✅ LangGraph completed in {graph_time:.2f}s")
        stage_times["langgraph"] = graph_time

        # Determine final source type based on ACTUAL usage
        if pdf_used and web_used:
            final_source_type = "mixed"
            logger.info("📚 FINAL: Mixed sources (PDF + Web)")
        elif pdf_used:
            final_source_type = "pdf"
            logger.info("📄 FINAL: PDF-only sources")
        elif web_used:
            final_source_type = "web"
            logger.info("🌐 FINAL: Web-only sources")
        else:
            final_source_type = source_type
            logger.info(f"⚠️ FINAL: No sources used (fallback to {source_type})")

        logger.info(f"📊 Source usage - PDF: {pdf_used}, Web: {web_used}, Final type: {final_source_type}")

        # ====================================================================
        # STEP 6: Generate streaming response (using your generator)
        # ====================================================================
        logger.info("📝 Starting response generation...")
        gen_start = time.time()
        
        # Prepare final state for generator
        final_state = {
            "question": resolved,
            "original_question": original_question,
            "refined_context": graph_results.get("refined_context", ""),
            "source_type": final_source_type,
            "source_urls": source_urls,
            "source_titles": source_titles
        }
        
        # Stream tokens using your existing generator
        full_answer = ""
        token_count = 0
        async for token in generate_stream(final_state, context):
            full_answer += token
            token_count += 1
            yield token
        
        # Emit image ref signals AFTER text answer — fetch from cloud URL as base64
        for image_url in image_refs_collected:
            b64 = fetch_image_as_base64(image_url)
            if b64:
                yield {"type": "image_ref", "data": b64}
        
        gen_time = time.time() - gen_start
        logger.info(f"✅ Generated {token_count} tokens in {gen_time:.2f}s")
        stage_times["generation"] = gen_time

        # ====================================================================
        # STEP 7: Add source footer - WITH CORRECT MIXED SOURCE SUPPORT!
        # ====================================================================
        if final_source_type == "web" and source_urls:
            # Only web sources
            footer = "\n\n" + "─" * 40 + "\n"
            footer += "🌐 **Sources (Web):**\n"
            for i, url in enumerate(source_urls[:3], 1):
                title = source_titles[i-1] if i-1 < len(source_titles) else "Web Source"
                footer += f"{i}. [{title}]({url})\n"
            yield footer
            full_answer += footer
            
        elif final_source_type == "pdf":
            # Only PDF sources
            footer = "\n\n" + "─" * 40 + "\n"
            footer += "📄 **Source: Uploaded PDF documents**"
            yield footer
            full_answer += footer
            
        elif final_source_type == "mixed":
            # BOTH PDF and Web sources were actually used!
            footer = "\n\n" + "─" * 40 + "\n"
            footer += "📚 **Combined Sources:**\n\n"
            
            # PDF section
            footer += "📄 **PDF Documents:**\n"
            footer += "   • Uploaded PDF files provided context\n\n"
            
            # Web section
            if source_urls:
                footer += "🌐 **Web Sources:**\n"
                for i, url in enumerate(source_urls[:3], 1):
                    title = source_titles[i-1] if i-1 < len(source_titles) else "Web Source"
                    footer += f"   {i}. [{title}]({url})\n"
            
            yield footer
            full_answer += footer

        # ====================================================================
        # STEP 8: Save to MongoDB and cache
        # ====================================================================
        if thread_id and full_answer:
            # Prepare sources for storage
            sources = []
            if source_urls:
                sources.extend([
                    {"url": url, "title": title, "type": "web"} 
                    for url, title in zip(source_urls[:5], source_titles[:5])
                ])
            if pdf_used:  # Only add PDF if it was actually used
                sources.append({"type": "pdf", "description": "Uploaded PDF documents"})
            
            # Save messages in background
            asyncio.create_task(save_message(
                thread_id=thread_id,
                role="user",
                content=original_question,
                tokens=len(original_question.split())
            ))
            
            asyncio.create_task(save_message(
                thread_id=thread_id,
                role="assistant",
                content=full_answer,
                tokens=len(full_answer.split()),
                sources=sources,
                metadata={
                    "verdict": graph_results.get("verdict"),
                    "from_cache": False,
                    "source_type": final_source_type,
                    "pdf_used": pdf_used,
                    "web_used": web_used
                }
            ))
            
            # Cache the answer in background
            asyncio.create_task(cache_set(
                thread_id=thread_id,
                question=resolved,
                answer=full_answer,
                metadata={
                    "source_type": final_source_type,
                    "source_urls": source_urls,
                    "source_titles": source_titles,
                    "pdf_used": pdf_used,
                    "web_used": web_used
                }
            ))
            
            logger.debug(f"Background save tasks created for thread {thread_id}")

        # ====================================================================
        # Log total performance
        # ====================================================================
        total_time = time.time() - overall_start
        logger.info(f"✅ TOTAL REQUEST TIME: {total_time:.2f}s")
        logger.debug(f"Stage times: {stage_times}")

    except Exception as e:
        logger.error(f"❌ Error in simple_stream_graph: {e}", exc_info=True)
        yield f"\n\n❌ Error: {str(e)}\n"

# Optional: Sync version for non-streaming (if needed)
async def simple_graph(inputs: dict) -> Dict:
    """
    Non-streaming version that returns the final state
    """
    full_answer = ""
    async for token in simple_stream_graph(inputs):
        full_answer += token
    
    return {"answer": full_answer}

# Performance monitoring endpoint (optional)
async def get_graph_stats() -> Dict:
    """Get statistics about the graph's performance"""
    return {
        "graph_type": "Hybrid (Custom + LangGraph)",
        "nodes": ["retrieve", "evaluate", "rewrite", "web_search", "refine"],
        "checkpointing": True,
        "streaming": True,
        "compiled": True
    }