import json
import random
import os
from typing import List, Dict, Any, Optional

class ConversationHandler:
    """Handles general conversational capabilities beyond ALU-specific knowledge"""
    
    def __init__(self, conversation_file_path="conversations.json"):
        self.conversation_guide = {}
        try:
            # Try to load conversation guide
            if os.path.exists(conversation_file_path):
                with open(conversation_file_path, "r", encoding="utf-8") as f:
                    self.conversation_guide = json.load(f)
                print(f"✅ Conversation guide loaded from {conversation_file_path}")
            else:
                print(f"⚠️ Conversation file not found at {conversation_file_path}")
        except Exception as e:
            print(f"⚠️ Error loading conversation guide: {e}")
    
    def get_random_response(self, category: str, default: str) -> str:
        """Get a random response from a specific category"""
        responses = self.conversation_guide.get("general_responses", {}).get(category, [default])
        return random.choice(responses)
    
    def detect_intent(self, message: str) -> Optional[str]:
        """Detect the user's intent from their message"""
        message_lower = message.lower()
        
        # Detect greetings
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "greetings", "morning", "afternoon", "evening"]):
            return "greeting"
            
        # Detect farewells
        if any(farewell in message_lower for farewell in ["bye", "goodbye", "see you", "farewell", "later"]):
            return "farewell"
            
        # Detect gratitude
        if any(thanks in message_lower for thanks in ["thanks", "thank you", "appreciate", "grateful"]):
            return "gratitude"
            
        # Check for casual topics
        for topic, content in self.conversation_guide.get("casual_topics", {}).items():
            topic_keywords = topic.split("_")
            if any(keyword in message_lower for keyword in topic_keywords):
                return f"casual_topic:{topic}"
                
        return None
    
    def get_response(self, message: str, conversation_history: List[Dict[str, str]]) -> str:
        """Generate a conversational response"""
        intent = self.detect_intent(message)
        
        if intent == "greeting":
            return self.get_random_response("greeting", "Hello! How can I help you with questions about ALU?")
            
        if intent == "farewell":
            return self.get_random_response("farewell", "Goodbye! Feel free to come back if you have more questions about ALU.")
            
        if intent == "gratitude":
            return self.get_random_response("gratitude", "You're welcome! Happy to help with your ALU questions.")
            
        if intent and intent.startswith("casual_topic:"):
            topic = intent.split(":", 1)[1]
            return self.conversation_guide.get("casual_topics", {}).get(topic, {}).get(
                "response", 
                "That's an interesting topic. How can I help with your ALU-related questions?"
            )
        
        # If no specific intent is detected, fall back to a clarification response
        return self.get_random_response(
            "clarification", 
            "I'm not sure I understood your question. Could you rephrase it so I can help you better?"
        )