"""
Personality and conversation style management for the chatbot
"""
import random
from typing import Dict, List, Optional

class ConversationPersonality:
    """Manages conversation personality and style"""
    
    @staticmethod
    def get_greeting_for_context(context: Dict) -> str:
        """Get appropriate greeting based on context"""
        msg_count = len(context.get("recent_messages", []))
        
        if msg_count == 0:
            greetings = [
                "Hey there! 👋 What can I help you with today?",
                "Hello! 😊 Ready to chat about anything that's on your mind.",
                "Hi! I'm here to help. What would you like to know?",
                "Welcome! 👋 I'm excited to assist you today."
            ]
        else:
            greetings = [
                "Welcome back! 👋 What's on your mind?",
                "Good to see you again! 😊 How can I help?",
                "Hey! Continuing where we left off?",
                "Back for more? Great! What would you like to explore?"
            ]
        
        return random.choice(greetings)
    
    @staticmethod
    def get_transition_phrases(last_topic: Optional[str] = None) -> str:
        """Get context-appropriate transition phrases"""
        if last_topic:
            phrases = [
                f"Building on what we discussed about {last_topic}...",
                f"Speaking of {last_topic}...",
                f"To add to our conversation about {last_topic}...",
                f"Since you mentioned {last_topic} earlier..."
            ]
        else:
            phrases = [
                "By the way,",
                "Interestingly,",
                "You might find this relevant...",
                "This connects to something fascinating..."
            ]
        
        return random.choice(phrases)
    
    @staticmethod
    def get_acknowledgment_phrases() -> str:
        """Get acknowledgment phrases"""
        phrases = [
            "That's a great question!",
            "Excellent question!",
            "I love this question!",
            "That's really interesting!",
            "Great point!",
            "I'm glad you asked!"
        ]
        return random.choice(phrases)
    
    @staticmethod
    def get_clarification_phrases() -> str:
        """Get clarification phrases"""
        phrases = [
            "Just to make sure I understand...",
            "Let me clarify...",
            "To be more precise...",
            "I want to make sure I'm addressing your question..."
        ]
        return random.choice(phrases)
    
    @staticmethod
    def get_follow_up_acknowledgment(depth: int) -> str:
        """Acknowledge follow-up questions based on depth"""
        if depth == 1:
            phrases = [
                "Great follow-up!",
                "Good question!",
                "Building on that..."
            ]
        elif depth == 2:
            phrases = [
                "Excellent follow-up!",
                "You're really digging deep - I like that!",
                "Great point to explore further!"
            ]
        else:
            phrases = [
                "This is getting interesting!",
                "What a great chain of questions!",
                "I love how you're exploring this topic!"
            ]
        return random.choice(phrases)
    
    @staticmethod
    def get_explanation_intro(tech_level: str) -> str:
        """Get explanation introduction based on technical level"""
        if tech_level == "beginner":
            phrases = [
                "Think of it this way...",
                "Here's a simple way to understand it:",
                "Let me break it down simply:"
            ]
        elif tech_level == "intermediate":
            phrases = [
                "Here's how it works:",
                "Let me explain the mechanics:",
                "The key concept is:"
            ]
        else:
            phrases = [
                "From a technical perspective,",
                "The architecture works like this:",
                "In technical terms,"
            ]
        return random.choice(phrases)