from typing import Any, Dict, AsyncGenerator, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging
import json

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)

def _build_conversation_stage_description(recent_messages: List[str], msg_count: int) -> str:
    """Determine conversation stage with personality"""
    if msg_count == 0:
        return {
            "stage": "🎯 Starting fresh",
            "description": "This is your first interaction - be warm, welcoming, and engaging",
            "style": "friendly and open-ended"
        }
    elif msg_count < 3:
        return {
            "stage": "🤝 Building rapport",
            "description": "Early conversation - show you're attentive and remember what they said",
            "style": "warm and curious"
        }
    elif msg_count < 8:
        return {
            "stage": "💬 Established conversation",
            "description": "You've built some rapport - can be more casual and reference shared context",
            "style": "conversational and connected"
        }
    else:
        return {
            "stage": "👥 Deep conversation",
            "description": "You know each other well - can make assumptions based on history",
            "style": "familiar and insightful"
        }

def _detect_conversation_flow(recent_messages: List[str]) -> Dict:
    """Analyze conversation flow for better responses"""
    if len(recent_messages) < 2:
        return {"type": "new", "follow_up_depth": 0}
    
    # Check if this is a follow-up
    last_user_msg = None
    for msg in reversed(recent_messages):
        if msg.startswith("User:"):
            last_user_msg = msg[5:].strip()
            break
    
    # Simple follow-up detection
    if last_user_msg and len(last_user_msg.split()) < 8:
        return {"type": "likely_follow_up", "depth": 1}
    
    return {"type": "new_topic", "depth": 0}

def _get_response_style_guidance(user_profile: Dict, flow: Dict) -> str:
    """Get style guidance based on user and conversation flow"""
    tech_level = user_profile.get("technical_level", "beginner")
    style = []
    
    # Technical level adaptation
    if tech_level == "beginner":
        style.append("• Use simple analogies and avoid jargon")
        style.append("• Explain technical terms when first used")
        style.append("• Be patient and offer to clarify if needed")
    elif tech_level == "intermediate":
        style.append("• Use appropriate terminology but explain complex concepts")
        style.append("• Provide examples that build on their knowledge")
    else:  # advanced
        style.append("• Feel free to use technical language")
        style.append("• Dive deep into details and edge cases")
    
    # Flow-based adaptation
    if flow.get("type") == "follow_up":
        style.append("• Acknowledge it's a follow-up and connect to previous answer")
        style.append("• Build upon what you just explained")
    
    return "\n".join(style)

def _build_system_prompt(context: Dict[str, Any], question: str) -> str:
    """
    Build a ChatGPT-style system prompt with rich context and personality
    """
    user_profile = context.get("user_profile", {})
    current_state = context.get("current_state", {})
    recent = context.get("recent_messages", [])
    facts = context.get("user_facts", [])
    summary = context.get("summary")
    analytics = context.get("analytics", {})
    
    # Format recent conversation beautifully
    conversation_flow = []
    for i, msg in enumerate(recent[-8:]):
        if msg.startswith("User:"):
            conversation_flow.append(f"👤 {msg}")
        elif msg.startswith("Assistant:"):
            conversation_flow.append(f"🤖 {msg}")
    
    recent_text = "\n".join(conversation_flow) if conversation_flow else "No previous conversation"

    # Format conversation summary (covers everything outside the recent window)
    summary = context.get("summary")
    summary_section = ""
    if summary:
        summary_section = f"""
## 🧠 CONVERSATION MEMORY (what happened before the recent messages)
{summary}

"""
    
    # Format user facts with emojis
    facts_text = ""
    if facts:
        fact_items = []
        for fact in facts[:3]:
            if "like" in fact.lower() or "love" in fact.lower():
                fact_items.append(f"❤️ {fact}")
            elif "work" in fact.lower() or "job" in fact.lower():
                fact_items.append(f"💼 {fact}")
            elif "learn" in fact.lower() or "study" in fact.lower():
                fact_items.append(f"📚 {fact}")
            else:
                fact_items.append(f"📌 {fact}")
        facts_text = "\n".join(fact_items)
    else:
        facts_text = "Still getting to know them"
    
    # Determine conversation stage with personality
    stage_info = _build_conversation_stage_description(recent, len(recent))
    flow = _detect_conversation_flow(recent)
    style_guidance = _get_response_style_guidance(user_profile, flow)
    
    # Sentiment analysis for tone adaptation
    sentiment = analytics.get("sentiment_trend", "neutral")
    sentiment_guidance = {
        "positive": "They seem engaged and positive - match their enthusiasm!",
        "negative": "They might be confused - be extra helpful and patient",
        "neutral": "Maintain a warm, professional tone"
    }.get(sentiment, "Be warm and natural")
    
    # Build the enhanced prompt
    prompt = f"""You are a brilliant, empathetic AI assistant with exceptional conversational skills.

## 🧠 CONVERSATION STAGE
{stage_info['stage']}
{stage_info['description']}
Style: {stage_info['style']}
{summary_section}
## 👤 ABOUT THE USER
• Technical level: {user_profile.get('technical_level', 'beginner')}
• Interests: {', '.join(user_profile.get('interests', [])[:5]) or 'Still discovering'}
• Current topic: {current_state.get('active_topic', 'Just getting started')}

## 💭 WHAT YOU KNOW ABOUT THEM
{facts_text}

## 📝 RECENT CONVERSATION
{recent_text}

## 🎨 RESPONSE STYLE GUIDANCE
{style_guidance}

## 🌡️ EMOTIONAL CONTEXT
{sentiment_guidance}

## 📚 KNOWLEDGE SOURCES
{_format_knowledge_sources(context)}

## ⚡ CONVERSATION PRINCIPLES
1. **Be Natural**: Sound like a knowledgeable friend, not a robot
2. **Show Memory**: Reference relevant parts of your conversation naturally — use the summary for older context
3. **Don't Repeat**: If something was already explained and is in the summary, build on it rather than re-explaining
4. **Build Bridges**: Connect current question to previous topics
5. **Add Personality**: Use occasional emojis, humor, and warmth
6. **Stay Relevant**: Keep responses focused and valuable
7. **Be Honest**: If unsure, admit it and suggest alternatives

## 🎯 EXAMPLE PHRASES FOR NATURAL CONVERSATION
• "Building on what we discussed about X..."
• "As you mentioned earlier..."
• "Since you're interested in Y, you might find this interesting..."
• "To add to that point about Z..."
• "Great question! This actually connects to..."

## ❓ CURRENT QUESTION
"{question}"

Now, respond naturally and brilliantly:"""
    
    return prompt

def _format_knowledge_sources(context: Dict) -> str:
    """Format available knowledge sources with emojis"""
    sources = []
    
    if context.get("has_pdf_docs"):
        sources.append("📄 Uploaded PDF documents")
    if context.get("has_web_sources"):
        sources.append("🌐 Web search results")
    if context.get("has_cache"):
        sources.append("⚡ Previously answered (cache)")
    
    return "\n".join(sources) if sources else "General knowledge"

async def generate_stream(state: Dict[str, Any], context: Optional[Dict] = None) -> AsyncGenerator[str, None]:
    """
    Enhanced streaming generator with personality and context awareness
    """
    question = state["question"]
    original_question = state.get("original_question", question)
    refined_context = state.get("refined_context", "")
    context = context or {}
    
    # Add source information to context
    context["has_pdf_docs"] = state.get("source_type") == "pdf" or state.get("pdf_used", False)
    context["has_web_sources"] = state.get("source_type") == "web" or state.get("web_used", False)
    context["has_cache"] = state.get("source_type") == "cache"
    
    # Build enhanced system prompt
    system_prompt = _build_system_prompt(context, original_question)
    
    # Add RAG context if available (with nice formatting)
    if refined_context:
        system_prompt += f"\n\n## 📖 RELEVANT INFORMATION FROM SOURCES\n{refined_context}"
    
    # Create messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)  # Use resolved question for RAG
    ]
    
    # Stream response
    full_response = ""
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
    except Exception as e:
        logger.error(f"Generation error: {e}")
        error_message = "\n\nI apologize, but I encountered an error. Let me try to help you differently. Could you rephrase your question?"
        yield error_message
        full_response += error_message
    
    # Store in state for later use
    state["full_response"] = full_response