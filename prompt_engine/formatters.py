
from typing import List, Dict, Any
from retrieval_engine import Document

class ContentFormatter:
    """Handles formatting of context and conversation history"""
    
    @staticmethod
    def format_context(documents: List[Document]) -> str:
        """Format context documents into a string"""
        if not documents:
            return "No relevant context found."
            
        formatted_context = "Here is relevant information from the ALU knowledge base:\n\n"
        
        for i, doc in enumerate(documents):
            formatted_context += f"Document {i+1}: {doc.metadata.get('title', 'Untitled')}\n"
            formatted_context += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            formatted_context += f"Content: {doc.text}\n\n"
            
        return formatted_context
    
    @staticmethod
    def format_conversation_history(conversation_history: List[Dict[str, Any]]) -> str:
        """Format conversation history into a string"""
        if not conversation_history:
            return "No previous conversation."
            
        formatted_history = "Previous messages:\n\n"
        
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("text", msg.get("content", ""))
            formatted_history += f"{role.capitalize()}: {content}\n\n"
            
        return formatted_history
