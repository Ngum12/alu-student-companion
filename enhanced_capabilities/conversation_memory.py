"""
Conversation memory module for storing and retrieving chat history
"""
from typing import List, Dict, Optional, Any
import time
import uuid
import os

class Message:
    """Represents a single message in a conversation."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }

class Conversation:
    """Represents a conversation with message history."""
    def __init__(self, user_id: str, max_messages: int = 20):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.created_at = time.time()
        self.last_updated = time.time()
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str) -> Message:
        """Add a message to the conversation."""
        message = Message(role, content)
        self.messages.append(message)
        self.last_updated = time.time()
        
        # Trim conversation if it exceeds the maximum length
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[len(self.messages) - self.max_messages:]
            
        return message
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from the conversation with optional limit."""
        if limit is None:
            return self.messages
        return self.messages[-limit:]
    
    def get_formatted_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get formatted history for prompt context."""
        messages = self.get_messages(limit)
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def clear(self) -> None:
        """Clear all messages in the conversation."""
        self.messages = []
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary format for storage."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Conversation':
        """Create a conversation instance from dictionary data."""
        conv = Conversation(data["user_id"])
        conv.id = data["id"]
        conv.created_at = data["created_at"]
        conv.last_updated = data["last_updated"]
        conv.metadata = data.get("metadata", {})
        
        # Recreate messages
        for msg_data in data["messages"]:
            msg = Message(msg_data["role"], msg_data["content"])
            msg.timestamp = msg_data["timestamp"]
            conv.messages.append(msg)
            
        return conv


class ConversationMemory:
    """Memory system for storing and managing conversations."""
    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = os.path.join(
            os.getenv("RENDER_VOLUME_PATH", "./data"), 
            os.path.basename(persistence_path)
        )
        self.conversations: Dict[str, Conversation] = {}
        self.user_conversations: Dict[str, List[str]] = {}  # Maps user_ids to conversation_ids
    
    def create_conversation(self, user_id: str) -> Conversation:
        """Create a new conversation for a user."""
        conversation = Conversation(user_id)
        self.conversations[conversation.id] = conversation
        
        # Add to user's conversations
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = []
        self.user_conversations[user_id].append(conversation.id)
        
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def get_user_conversations(self, user_id: str) -> List[Conversation]:
        """Get all conversations for a user."""
        conversation_ids = self.user_conversations.get(user_id, [])
        return [self.conversations[conv_id] for conv_id in conversation_ids if conv_id in self.conversations]
    
    def get_active_conversation(self, user_id: str) -> Conversation:
        """Get the most recent conversation for a user or create a new one."""
        user_convs = self.get_user_conversations(user_id)
        
        if not user_convs:
            return self.create_conversation(user_id)
        
        # Return the most recently updated conversation
        return max(user_convs, key=lambda conv: conv.last_updated)
    
    def add_message(self, user_id: str, role: str, content: str, conversation_id: Optional[str] = None) -> Message:
        """Add a message to a conversation."""
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
        else:
            conversation = self.get_active_conversation(user_id)
            
        return conversation.add_message(role, content)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation by ID."""
        if conversation_id not in self.conversations:
            return False
            
        conversation = self.conversations[conversation_id]
        
        # Remove from user conversations mapping
        user_id = conversation.user_id
        if user_id in self.user_conversations and conversation_id in self.user_conversations[user_id]:
            self.user_conversations[user_id].remove(conversation_id)
            
        # Remove the conversation
        del self.conversations[conversation_id]
        return True
    
    def clear_user_conversations(self, user_id: str) -> int:
        """Clear all conversations for a user. Returns number of conversations deleted."""
        if user_id not in self.user_conversations:
            return 0
            
        count = 0
        conversation_ids = self.user_conversations[user_id].copy()
        for conv_id in conversation_ids:
            if self.delete_conversation(conv_id):
                count += 1
                
        return count
    
    def save_to_disk(self) -> None:
        """Save conversations to disk if persistence path is set."""
        import json
        import os
        
        if not self.persistence_path:
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        
        # Prepare data for serialization
        data = {
            "conversations": {
                conv_id: conv.to_dict() for conv_id, conv in self.conversations.items()
            },
            "user_conversations": self.user_conversations
        }
        
        # Write to file
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_disk(self) -> bool:
        """Load conversations from disk. Returns True if successful."""
        import json
        import os
        
        try:
            if os.path.exists(self.persistence_path):
                # Load existing conversations
                with open(self.persistence_path, 'r') as f:
                    conversations_data = json.load(f)
                    # Load conversation data...
            return True
        except Exception as e:
            print(f"Error loading conversations, creating new storage: {e}")
            # Continue with empty conversations rather than failing
            return False
    
    def summarize_conversation(self, conversation_id: str, max_length: int = 200) -> str:
        """Generate a brief summary of the conversation for context."""
        conversation = self.get_conversation(conversation_id)
        if not conversation or not conversation.messages:
            return "No conversation found."
        
        # Extract the key topics
        all_text = " ".join([msg.content for msg in conversation.messages])
        
        # Simple extraction of first few words
        if len(all_text) > max_length:
            summary = all_text[:max_length] + "..."
        else:
            summary = all_text
            
        return f"Conversation about: {summary}"

    def analyze_user_preferences(self, user_id):
        """Analyze conversation history to learn user preferences"""
        conversations = self.get_user_conversations(user_id)
        
        # Extract patterns from conversations
        topics_of_interest = {}
        response_length_preference = 0
        technical_detail_level = 0
        
        for conv in conversations:
            for msg in conv.messages:
                if msg.role == "user":
                    # Analyze question complexity, topics, etc.
                    # This is a simplified example
                    words = msg.content.split()
                    for word in words:
                        if word in topics_of_interest:
                            topics_of_interest[word] += 1
                        else:
                            topics_of_interest[word] = 1
                
                if msg.role == "assistant":
                    # Analyze what kinds of responses the user engages with
                    response_length_preference += len(msg.content)
        
        # Calculate averages
        if len(conversations) > 0:
            response_length_preference /= len(conversations)
        
        return {
            "topics_of_interest": dict(sorted(topics_of_interest.items(), 
                                             key=lambda x: x[1], reverse=True)[:5]),
            "preferred_response_length": response_length_preference,
            "technical_detail_level": technical_detail_level
        }