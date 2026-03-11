"""
MongoDB Atlas Unified Manager
Replaces: SQLite (conversations) + Redis (cache)
Provides: ChatGPT-level conversation memory + intelligent caching
"""

import os
import hashlib
import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_openai import OpenAIEmbeddings
from bson import ObjectId
from dotenv import load_dotenv 
load_dotenv()   

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

THREAD_CACHE_TTL = 6 * 60 * 60  # 6 hours
SIMILARITY_THRESHOLD = 0.85  # Lowered slightly for better recall
MAX_SEMANTIC_ENTRIES = 200  # Keep newest 200 per thread

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ============================================================================
# MONGODB CONNECTION (Singleton)
# ============================================================================

class MongoDBConnection:
    """Singleton for MongoDB Atlas connection"""
    _instance = None
    _client = None
    _async_client = None
    _db = None
    _async_db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connect()
        return cls._instance

    def _connect(self):
        """Establish connection to MongoDB Atlas"""
        # mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI environment variable not set!")

        # Sync client
        self._client = MongoClient(mongodb_uri, maxPoolSize=50)
        self._db = self._client['crag']
        
        # Async client for streaming
        self._async_client = AsyncIOMotorClient(mongodb_uri, maxPoolSize=50)
        self._async_db = self._async_client['crag']
        
        self._init_indexes()
        logger.info("✅ MongoDB Atlas connected")

    def _init_indexes(self):
        """Create all necessary indexes for optimal performance"""
        
        # Threads collection
        self._db.threads.create_index("thread_id", unique=True)
        self._db.threads.create_index([("last_active", DESCENDING)])
        self._db.threads.create_index("user_profile.interests")
        
        # Messages collection
        self._db.messages.create_index([("thread_id", ASCENDING), ("created_at", DESCENDING)])
        self._db.messages.create_index("topics")
        self._db.messages.create_index("sentiment")
        
        # Exact cache - thread-specific
        self._db.cache_exact.create_index(
            [("thread_id", ASCENDING), ("question_hash", ASCENDING)],
            unique=True
        )
        self._db.cache_exact.create_index("expires_at", expireAfterSeconds=0)
        
        # Semantic cache - thread-specific
        self._db.cache_semantic.create_index([("thread_id", ASCENDING), ("created_at", DESCENDING)])
        self._db.cache_semantic.create_index("expires_at", expireAfterSeconds=0)
        
        # User facts
        self._db.user_facts.create_index([("thread_id", ASCENDING), ("fact_hash", ASCENDING)])
        self._db.user_facts.create_index("last_mentioned")
        
        # Summaries
        self._db.summaries.create_index([("thread_id", ASCENDING), ("created_at", DESCENDING)])

    @property
    def db(self):
        return self._db

    @property
    def async_db(self):
        return self._async_db

# Initialize connection
_mongo = MongoDBConnection()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between vectors"""
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def _get_question_hash(question: str) -> str:
    """Generate deterministic hash for question"""
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]

def _extract_topics(text: str) -> List[str]:
    """Extract topics from text"""
    common_topics = [
        "weather", "news", "technology", "science", "sports", "music", "movies",
        "books", "travel", "food", "health", "business", "education", "programming",
        "AI", "data", "politics", "economy", "environment", "art", "history",
        "philosophy", "psychology", "medicine", "engineering", "RAG", "vectors",
        "databases", "scaling", "performance", "optimization"
    ]
    text_lower = text.lower()
    return [t for t in common_topics if t in text_lower]

def _analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text"""
    positive = ['good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful',
                'fantastic', 'amazing', 'perfect', 'thanks', 'helpful']
    negative = ['bad', 'terrible', 'awful', 'hate', 'worst', 'sad', 'angry',
                'horrible', 'confusing', 'wrong', 'error', 'fail']
    
    text_lower = text.lower()
    pos_count = sum(1 for w in positive if w in text_lower)
    neg_count = sum(1 for w in negative if w in text_lower)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"

def _classify_question(question: str) -> str:
    """Classify question type"""
    question_lower = question.lower()
    
    if question_lower.startswith(('what', 'who', 'when', 'where')):
        return "definition"
    elif question_lower.startswith(('how', 'can you show', 'example')):
        return "procedural"
    elif question_lower.startswith(('why', 'explain')):
        return "explanatory"
    elif ' vs ' in question_lower or ' versus ' in question_lower or 'compare' in question_lower:
        return "comparison"
    elif question_lower.startswith(('is', 'are', 'do', 'does', 'can')):
        return "verification"
    elif len(question_lower.split()) < 4:
        return "follow_up"
    else:
        return "general"

# ============================================================================
# THREAD MANAGEMENT
# ============================================================================

def create_thread(thread_id: str, title: Optional[str] = None) -> Dict:
    """Create a new conversation thread"""
    thread = {
        "thread_id": thread_id,
        "title": title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": datetime.utcnow(),
        "last_active": datetime.utcnow(),
        "message_count": 0,
        "user_profile": {
            "technical_level": "beginner",
            "preferred_response_style": "balanced",
            "interests": [],
            "communication_style": "unknown",
            "known_concepts": []
        },
        "current_state": {
            "active_topic": None,
            "explanation_depth": "basic",
            "pending_clarification": False,
            "last_question_type": None,
            "follow_up_count": 0
        },
        "analytics": {
            "avg_response_length": 0,
            "common_question_types": {},
            "sentiment_trend": "neutral",
            "topic_switches": 0
        },
        "topics": [],
        "summary": None
    }
    
    try:
        _mongo.db.threads.insert_one(thread)
        logger.info(f"Thread created: {thread_id}")
        return thread
    except DuplicateKeyError:
        logger.warning(f"Thread {thread_id} already exists")
        return get_thread_metadata(thread_id)

def thread_exists(thread_id: str) -> bool:
    """Check if thread exists"""
    return _mongo.db.threads.count_documents({"thread_id": thread_id}) > 0

def get_thread_metadata(thread_id: str) -> Dict:
    """Get thread metadata"""
    thread = _mongo.db.threads.find_one({"thread_id": thread_id})
    if thread:
        return {
            "thread_id": thread["thread_id"],
            "title": thread.get("title", ""),
            "message_count": thread.get("message_count", 0),
            "topics": thread.get("topics", []),
            "summary": thread.get("summary"),
            "created_at": thread.get("created_at"),
            "last_active": thread.get("last_active"),
            "user_profile": thread.get("user_profile", {}),
            "current_state": thread.get("current_state", {})
        }
    return {}

def update_thread_title(thread_id: str, title: str):
    """Update thread title"""
    _mongo.db.threads.update_one(
        {"thread_id": thread_id},
        {"$set": {"title": title}}
    )

def get_recent_threads(limit: int = 10) -> List[Dict]:
    """Get recent threads for sidebar"""
    threads = _mongo.db.threads.find().sort("last_active", DESCENDING).limit(limit)
    
    result = []
    for t in threads:
        # Get last message preview
        last_msg = _mongo.db.messages.find_one(
            {"thread_id": t["thread_id"]},
            sort=[("created_at", DESCENDING)]
        )
        
        result.append({
            "thread_id": t["thread_id"],
            "title": t.get("title", ""),
            "last_active": t.get("last_active").isoformat(),
            "message_count": t.get("message_count", 0),
            "topics": t.get("topics", [])[:3],
            "preview": last_msg.get("content", "")[:100] if last_msg else ""
        })
    
    return result

def delete_thread(thread_id: str):
    """Delete thread and all associated data"""
    _mongo.db.user_facts.delete_many({"thread_id": thread_id})
    _mongo.db.summaries.delete_many({"thread_id": thread_id})
    _mongo.db.messages.delete_many({"thread_id": thread_id})
    _mongo.db.cache_exact.delete_many({"thread_id": thread_id})
    _mongo.db.cache_semantic.delete_many({"thread_id": thread_id})
    _mongo.db.threads.delete_one({"thread_id": thread_id})
    logger.info(f"Thread {thread_id} deleted")

# ============================================================================
# MESSAGE MANAGEMENT
# ============================================================================

async def save_message(thread_id: str, role: str, content: str, tokens: int = 0,
                       metadata: Optional[Dict] = None, sources: Optional[List[Dict]] = None):
    """
    Save a message with rich metadata
    """
    db = _mongo.async_db
    
    # Extract topics and sentiment
    topics = _extract_topics(content)
    sentiment = _analyze_sentiment(content)
    
    # Create message document
    message = {
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "tokens": tokens,
        "topics": topics,
        "sentiment": sentiment,
        "created_at": datetime.utcnow()
    }
    
    # Add response metadata for assistant messages
    if role == "assistant" and sources:
        message["response_metadata"] = {
            "sources": sources,
            "from_cache": metadata.get("from_cache", False) if metadata else False
        }
    
    # Get parent message ID for threading
    last_msg = await db.messages.find_one(
        {"thread_id": thread_id},
        sort=[("created_at", DESCENDING)]
    )
    if last_msg:
        message["parent_id"] = last_msg["_id"]
    
    # Insert message
    result = await db.messages.insert_one(message)
    
    # Update thread stats
    await db.threads.update_one(
        {"thread_id": thread_id},
        {
            "$set": {"last_active": datetime.utcnow()},
            "$inc": {"message_count": 1},
            "$addToSet": {"topics": {"$each": topics}}
        }
    )
    
    # If user message, extract facts and update profile
    if role == "user":
        await _extract_and_save_facts(thread_id, content)
        await _update_user_profile(thread_id, content)
    
    # Generate summary every 10 messages
    thread = await db.threads.find_one({"thread_id": thread_id})
    if thread and thread.get("message_count", 0) % 10 == 0:
        asyncio.create_task(_generate_summary(thread_id))
    
    return str(result.inserted_id)

async def load_memory(thread_id: str, limit: int = 50) -> List[Dict]:
    """Load conversation history with context"""
    db = _mongo.async_db
    
    cursor = db.messages.find(
        {"thread_id": thread_id}
    ).sort("created_at", ASCENDING).limit(limit)
    
    messages = []
    async for msg in cursor:
        m = {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg["created_at"].isoformat(),
            "tokens": msg.get("tokens", 0),
            "topics": msg.get("topics", [])
        }
        
        if msg.get("response_metadata"):
            m["sources"] = msg["response_metadata"].get("sources", [])
        
        messages.append(m)
    
    return messages

async def _extract_and_save_facts(thread_id: str, message: str):
    """Extract user facts from message"""
    db = _mongo.async_db
    
    # Patterns that indicate personal facts
    patterns = [
        "I am", "I'm", "my name", "I work", "I live", "I like",
        "I love", "I hate", "I prefer", "I have", "I need",
        "I study", "I learned", "I want", "I don't like",
        "my job", "my project", "my company"
    ]
    
    for pattern in patterns:
        if pattern.lower() in message.lower():
            # Extract sentence containing pattern
            sentences = message.split('.')
            for sentence in sentences:
                if pattern.lower() in sentence.lower():
                    fact = sentence.strip()
                    if len(fact) > 10:
                        fact_hash = hashlib.md5(fact.encode()).hexdigest()
                        
                        await db.user_facts.update_one(
                            {"thread_id": thread_id, "fact_hash": fact_hash},
                            {
                                "$set": {
                                    "fact": fact,
                                    "last_mentioned": datetime.utcnow()
                                },
                                "$inc": {"mentioned_count": 1},
                                "$setOnInsert": {
                                    "first_mentioned": datetime.utcnow(),
                                    "category": "personal"
                                }
                            },
                            upsert=True
                        )

async def _update_user_profile(thread_id: str, message: str):
    """Update user profile based on interactions"""
    db = _mongo.async_db
    
    # Get current profile
    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        return
    
    profile = thread.get("user_profile", {})
    
    # Update interests based on topics
    topics = _extract_topics(message)
    current_interests = profile.get("interests", [])
    profile["interests"] = list(set(current_interests + topics))
    
    # Determine technical level based on vocabulary
    advanced_terms = ['algorithm', 'architecture', 'optimization', 'scalability',
                      'implementation', 'vector', 'embedding', 'latency']
    tech_count = sum(1 for term in advanced_terms if term in message.lower())
    
    if tech_count >= 3:
        profile["technical_level"] = "advanced"
    elif tech_count >= 1:
        if profile.get("technical_level") == "beginner":
            profile["technical_level"] = "intermediate"
    
    # Update question type patterns
    q_type = _classify_question(message)
    analytics = thread.get("analytics", {})
    q_types = analytics.get("common_question_types", {})
    q_types[q_type] = q_types.get(q_type, 0) + 1
    analytics["common_question_types"] = q_types
    
    # Save updates
    await db.threads.update_one(
        {"thread_id": thread_id},
        {
            "$set": {
                "user_profile": profile,
                "analytics": analytics
            }
        }
    )

async def _generate_summary(thread_id: str):
    """
    Generate a structured, LLM-powered conversation summary every N messages.
    The summary is designed to be injected into the generator's system prompt,
    giving it real context about what was discussed, decided, and learned.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field as PydanticField

    db = _mongo.async_db

    # Fetch last 20 messages in chronological order
    cursor = db.messages.find(
        {"thread_id": thread_id}
    ).sort("created_at", DESCENDING).limit(20)

    raw_messages = []
    async for msg in cursor:
        raw_messages.append(msg)

    if len(raw_messages) < 5:
        return

    # Reverse so messages are in chronological order for the LLM
    raw_messages.reverse()

    # Format transcript
    transcript_lines = []
    for msg in raw_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate very long assistant messages to avoid token bloat
        content = msg["content"]
        if role == "Assistant" and len(content) > 500:
            content = content[:500] + "…"
        transcript_lines.append(f"{role}: {content}")
    transcript = "\n".join(transcript_lines)

    # Structured output schema
    class ConversationSummary(BaseModel):
        one_line: str = PydanticField(
            description="One sentence capturing the core topic and goal of this conversation."
        )
        what_was_covered: list[str] = PydanticField(
            description="3-5 bullet points of the main concepts, questions, or tasks addressed so far."
        )
        conclusions_and_decisions: list[str] = PydanticField(
            description="Key conclusions reached, decisions made, or answers given. Empty list if none."
        )
        open_threads: list[str] = PydanticField(
            description="Unresolved questions, pending tasks, or topics the user seemed to want more on. Empty list if none."
        )
        user_context: str = PydanticField(
            description="1-2 sentences about what the user is trying to accomplish overall, their apparent level, and any personal context they shared."
        )
        suggested_next_topics: list[str] = PydanticField(
            description="2-3 natural follow-up topics or questions the user might ask next, based on the conversation arc."
        )

    _summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    _summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert conversation analyst. 
Read the transcript below and produce a structured summary that will help an AI assistant 
continue the conversation intelligently — as if it has perfect memory of everything discussed.

Be specific and concrete. Don't write vague phrases like "various topics were discussed."
Reference actual concepts, questions, and answers from the transcript.""",
        ),
        ("human", "Conversation transcript:\n\n{transcript}"),
    ])

    _summary_chain = _summary_prompt | _summary_llm.with_structured_output(
        ConversationSummary, include_raw=False
    )

    try:
        result: ConversationSummary = await _summary_chain.ainvoke({"transcript": transcript})

        # Build a human-readable narrative for injection into the generator prompt
        narrative_parts = [f"**Summary:** {result.one_line}"]

        if result.what_was_covered:
            narrative_parts.append(
                "**Covered so far:**\n" + "\n".join(f"• {p}" for p in result.what_was_covered)
            )

        if result.conclusions_and_decisions:
            narrative_parts.append(
                "**Conclusions / Decisions:**\n" + "\n".join(f"• {c}" for c in result.conclusions_and_decisions)
            )

        if result.open_threads:
            narrative_parts.append(
                "**Still open:**\n" + "\n".join(f"• {o}" for o in result.open_threads)
            )

        narrative_parts.append(f"**User context:** {result.user_context}")

        if result.suggested_next_topics:
            narrative_parts.append(
                "**Likely next questions:**\n" + "\n".join(f"• {s}" for s in result.suggested_next_topics)
            )

        narrative = "\n\n".join(narrative_parts)

        summary_doc = {
            "thread_id": thread_id,
            # Human-readable narrative for the generator prompt
            "summary": narrative,
            # Structured fields for analytics / future use
            "one_line": result.one_line,
            "what_was_covered": result.what_was_covered,
            "conclusions_and_decisions": result.conclusions_and_decisions,
            "open_threads": result.open_threads,
            "user_context": result.user_context,
            "suggested_next_topics": result.suggested_next_topics,
            "message_count": len(raw_messages),
            "created_at": datetime.utcnow(),
        }

        await db.summaries.insert_one(summary_doc)

        # Update thread with the rich narrative so get_conversation_context picks it up
        await db.threads.update_one(
            {"thread_id": thread_id},
            {"$set": {"summary": narrative}}
        )

        logger.info(f"📝 LLM summary generated for thread {thread_id[:8]}: {result.one_line}")

    except Exception as e:
        logger.warning(f"Summary generation failed for thread {thread_id[:8]}: {e}")

async def get_conversation_context(thread_id: str) -> Dict:
    """Get rich conversation context for the LLM"""
    db = _mongo.async_db
    
    # Get thread with all related data in one aggregation
    pipeline = [
        {"$match": {"thread_id": thread_id}},
        {"$lookup": {
            "from": "messages",
            "let": {"thread_id": "$thread_id"},
            "pipeline": [
                {"$match": {"$expr": {"$eq": ["$thread_id", "$$thread_id"]}}},
                {"$sort": {"created_at": -1}},
                {"$limit": 10},
                {"$project": {
                    "role": 1,
                    "content": 1,
                    "created_at": 1,
                    "topics": 1,
                    "sentiment": 1
                }}
            ],
            "as": "recent_messages"
        }},
        {"$lookup": {
            "from": "user_facts",
            "localField": "thread_id",
            "foreignField": "thread_id",
            "as": "user_facts"
        }},
        {"$lookup": {
            "from": "summaries",
            "localField": "thread_id",
            "foreignField": "thread_id",
            "as": "summaries"
        }}
    ]
    
    try:
        result = await db.threads.aggregate(pipeline).next()
    except:
        # Thread doesn't exist or has no data
        return {
            "thread_id": thread_id,
            "user_profile": {},
            "current_state": {},
            "recent_messages": [],
            "user_facts": [],
            "summary": None,
            "topics": [],
            "analytics": {}
        }
    
    # Format messages for prompt
    recent_formatted = []
    for msg in result.get("recent_messages", []):
        role = "User" if msg["role"] == "user" else "Assistant"
        recent_formatted.append(f"{role}: {msg['content']}")
    
    return {
        "thread_id": thread_id,
        "user_profile": result.get("user_profile", {}),
        "current_state": result.get("current_state", {}),
        "recent_messages": recent_formatted,
        "user_facts": [f["fact"] for f in result.get("user_facts", [])[:5]],
        "summary": result.get("summaries", [{}])[0].get("summary") if result.get("summaries") else None,
        "topics": result.get("topics", []),
        "analytics": result.get("analytics", {})
    }

# ============================================================================
# CACHE MANAGEMENT - COMPLETELY THREAD-ISOLATED WITH WORKING SEMANTIC CACHE
# ============================================================================

async def cache_get(thread_id: str, question: str) -> Optional[Tuple[str, bool]]:
    """
    Get cached answer with semantic matching - THREAD ISOLATED
    Returns (answer, from_semantic) or None
    """
    db = _mongo.async_db
    q = question.strip()
    q_hash = _get_question_hash(q)
    
    logger.info(f"🔍 Cache lookup for thread {thread_id[:8]}: '{q[:50]}...'")
    
    try:
        # 1. Exact match (fastest) - THREAD SPECIFIC ONLY
        exact = await db.cache_exact.find_one({
            "thread_id": thread_id,
            "question_hash": q_hash,
            "expires_at": {"$gt": datetime.utcnow()}
        })
        
        if exact:
            logger.info(f"🟢 Cache HIT (exact) for thread {thread_id[:8]}: '{q[:50]}...'")
            # Update hit count
            await db.cache_exact.update_one(
                {"_id": exact["_id"]},
                {"$inc": {"hit_count": 1}, "$set": {"last_accessed": datetime.utcnow()}}
            )
            return exact["answer"], False
        
        # 2. Semantic match - Try vector search first
        try:
            # Get query embedding
            q_emb = await asyncio.to_thread(_embeddings.embed_query, q)
            
            # Try brute force search (more reliable)
            cursor = db.cache_semantic.find({
                "thread_id": thread_id,
                "expires_at": {"$gt": datetime.utcnow()}
            })
            
            best_score = -1.0
            best_answer = None
            best_entry = None
            
            # Convert cursor to list for iteration
            entries = await cursor.to_list(length=100)
            
            for entry in entries:
                if "embedding" in entry:
                    score = _cosine_similarity(q_emb, entry["embedding"])
                    if score > best_score:
                        best_score = score
                        best_answer = entry["answer"]
                        best_entry = entry
            
            if best_score >= SIMILARITY_THRESHOLD:
                logger.info(f"🟡 Cache HIT (semantic, score={best_score:.3f}) for thread {thread_id[:8]}")
                
                # Update access stats
                if best_entry:
                    await db.cache_semantic.update_one(
                        {"_id": best_entry["_id"]},
                        {"$inc": {"access_count": 1}, "$set": {"last_accessed": datetime.utcnow()}}
                    )
                
                return best_answer, True
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
        
        logger.info(f"⚪ Cache MISS for thread {thread_id[:8]}: '{q[:50]}...'")
        return None
        
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        return None

async def cache_set(thread_id: str, question: str, answer: str, 
                   metadata: Optional[Dict] = None):
    """Store answer in cache - THREAD ISOLATED"""
    db = _mongo.async_db
    q = question.strip()
    q_hash = _get_question_hash(q)
    
    try:
        # Get embedding
        q_emb = await asyncio.to_thread(_embeddings.embed_query, q)
        
        # Calculate expiry
        expires_at = datetime.utcnow() + timedelta(seconds=THREAD_CACHE_TTL)
        
        # 1. Thread-scoped exact cache
        await db.cache_exact.update_one(
            {"thread_id": thread_id, "question_hash": q_hash},
            {
                "$set": {
                    "question": q,
                    "answer": answer,
                    "expires_at": expires_at,
                    "last_accessed": datetime.utcnow()
                },
                "$setOnInsert": {
                    "created_at": datetime.utcnow(),
                    "hit_count": 0
                }
            },
            upsert=True
        )
        
        # 2. Thread-scoped semantic cache
        await db.cache_semantic.insert_one({
            "thread_id": thread_id,
            "question": q,
            "answer": answer,
            "embedding": q_emb,
            "expires_at": expires_at,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 0,
            "metadata": metadata or {}
        })
        
        # Keep only newest MAX_SEMANTIC_ENTRIES for this thread
        count = await db.cache_semantic.count_documents({"thread_id": thread_id})
        if count > MAX_SEMANTIC_ENTRIES:
            # Find oldest entries to delete
            oldest = db.cache_semantic.find(
                {"thread_id": thread_id}
            ).sort("created_at", ASCENDING).limit(count - MAX_SEMANTIC_ENTRIES)
            
            async for old in oldest:
                await db.cache_semantic.delete_one({"_id": old["_id"]})
        
        logger.info(f"💾 Cached answer for thread {thread_id[:8]}: {q[:50]}...")
        
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")

def cache_invalidate_thread(thread_id: str):
    """Invalidate all cache entries for a specific thread"""
    result_exact = _mongo.db.cache_exact.delete_many({"thread_id": thread_id})
    result_semantic = _mongo.db.cache_semantic.delete_many({"thread_id": thread_id})
    logger.info(f"🗑️ Invalidated cache for thread {thread_id[:8]} (exact: {result_exact.deleted_count}, semantic: {result_semantic.deleted_count})")

async def get_cache_stats(thread_id: str) -> Dict:
    """Get cache statistics for a specific thread"""
    db = _mongo.async_db
    
    exact_count = await db.cache_exact.count_documents({"thread_id": thread_id})
    semantic_count = await db.cache_semantic.count_documents({"thread_id": thread_id})
    
    # Get most accessed
    pipeline = [
        {"$match": {"thread_id": thread_id}},
        {"$sort": {"access_count": -1}},
        {"$limit": 5},
        {"$project": {"question": 1, "access_count": 1, "created_at": 1}}
    ]
    
    top_accessed = []
    cursor = db.cache_semantic.aggregate(pipeline)
    async for item in cursor:
        top_accessed.append({
            "question": item["question"][:50],
            "access_count": item.get("access_count", 0),
            "created": item["created_at"].isoformat()
        })
    
    return {
        "thread_id": thread_id[:8],
        "exact_entries": exact_count,
        "semantic_entries": semantic_count,
        "total_entries": exact_count + semantic_count,
        "top_accessed": top_accessed
    }

# ============================================================================
# USER FACTS RETRIEVAL
# ============================================================================

async def get_user_facts(thread_id: str, min_mentions: int = 1) -> List[str]:
    """Get important facts about user"""
    db = _mongo.async_db
    
    facts = db.user_facts.find(
        {"thread_id": thread_id, "mentioned_count": {"$gte": min_mentions}}
    ).sort("mentioned_count", DESCENDING).limit(10)
    
    return [f["fact"] async for f in facts]

# ============================================================================
# FEEDBACK MANAGEMENT
# ============================================================================

async def save_feedback(thread_id: str, message_id: str, rating: int, feedback: str = None):
    """Save user feedback on responses"""
    db = _mongo.async_db
    
    await db.feedback.insert_one({
        "thread_id": thread_id,
        "message_id": ObjectId(message_id),
        "rating": rating,
        "feedback": feedback,
        "created_at": datetime.utcnow()
    })
    
    logger.info(f"Feedback saved for message {message_id}")
