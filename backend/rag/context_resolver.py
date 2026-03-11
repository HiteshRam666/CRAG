"""
Resolves ambiguous / follow-up questions into self-contained queries
using conversation history — enabling cache hits on semantically identical questions.
"""

import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_resolve_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert conversation analyst who excels at understanding context and resolving ambiguous references.

## YOUR TASK
Rewrite the user's LATEST question into a completely self-contained question that captures their true intent, even without seeing the conversation history.

## CONTEXT YOU HAVE

**Conversation Summary (everything discussed so far):**
{summary}

**Recent Messages (last few exchanges):**
{history}

**User Profile:**
- Interests: {interests}
- Current topic: {current_topic}
- Technical level: {tech_level}
- Key facts about user: {user_facts}

## THINKING PROCESS (internal - don't output)
1. What pronouns need resolution? (it, they, this, that, those)
2. What's the last mentioned entity that "it" could refer to?
3. Are there any vague references like "the algorithm" or "that concept"?
4. Is the user building on something mentioned earlier?
5. Check the summary — was this topic covered before, even outside the recent messages?
6. Would someone reading this question in isolation understand it?

## RULES
- If the question is already self-contained, return it EXACTLY as is
- Replace pronouns with the actual entities from conversation
- Expand vague references into explicit concepts — use the summary for older context
- Maintain the user's original intent and tone
- DO NOT add information that wasn't implied
- DO NOT answer the question - only rewrite it
- Output ONLY the rewritten question, nothing else

## EXAMPLES

Example 1:
History: User: "What is gradient descent?" Assistant: [explains gradient descent]
Follow-up: "Can you explain it more slowly?"
Rewritten: "Can you explain gradient descent more slowly?"

Example 2:
History: User: "I'm building a chatbot for customer service" Assistant: "Great! Let me help..."
Follow-up: "What's the best architecture?"
Rewritten: "What's the best architecture for a customer service chatbot?"

Example 3:
Summary: Covered hybrid BM25 + vector search, concluded k=6 is optimal
History: [only recent messages visible]
Follow-up: "What about the other approach we discussed?"
Rewritten: "What are the tradeoffs of BM25-only search compared to the hybrid approach we discussed?"

Now rewrite this question: {question}""",
    )
])

_resolve_chain = _resolve_prompt | _llm

async def resolve_question(
        question: str, 
        recent_messages: List[str],
        user_profile: Optional[Dict] = None
) -> str:
    """
    Resolve question using conversation history, summary, and user context.
    The summary covers everything outside the recent_messages window,
    enabling correct disambiguation even in long conversations.
    """
    # Quick heuristic: only resolve if needed
    followup_signals = [
        "it", "this", "that", "they", "them", "those", "these",
        "the above", "the same", "more", "another", "also",
        "what about", "how about", "explain more", "elaborate",
        "tell me more", "go deeper", "specifically", "exactly"
    ]

    q_lower = question.lower()
    needs_resolution = any(signal in q_lower for signal in followup_signals)

    # Short questions are almost always follow-ups
    if len(question.split()) <= 4:
        needs_resolution = True
    
    if not needs_resolution and len(question.split()) > 5:
        return question
    
    try:
        # Format recent messages window
        history = "\n".join(recent_messages[-8:]) if recent_messages else "No recent messages"

        # Extract summary — this covers everything outside the recent window
        summary = ""
        if user_profile:
            raw_summary = user_profile.get("summary") or user_profile.get("current_state", {}).get("summary")
            if raw_summary:
                summary = raw_summary
        if not summary:
            summary = "No summary yet — conversation is still early."

        # Get user context
        interests = ", ".join(user_profile.get("interests", [])[:3]) if user_profile else "unknown"
        current_topic = user_profile.get("current_state", {}).get("active_topic") if user_profile else "unknown"
        tech_level = user_profile.get("technical_level", "beginner") if user_profile else "beginner"
        user_facts = user_profile.get("facts", []) if user_profile else []
        facts_str = ", ".join(user_facts[:2]) if user_facts else "no known facts"
        
        # Call LLM
        result = await _resolve_chain.ainvoke({
            "summary": summary,
            "history": history,
            "interests": interests,
            "current_topic": current_topic,
            "tech_level": tech_level,
            "user_facts": facts_str,
            "question": question
        })
        
        resolved = result.content.strip()
        
        if resolved and resolved != question:
            logger.info(f"🔄 Resolved: '{question}' → '{resolved}'")
            return resolved
        
        return question
        
    except Exception as e:
        logger.warning(f"Resolution failed: {e}")
        return question