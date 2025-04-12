# Create a new file - analytics/conversation_analytics.py

from collections import Counter
from typing import Dict, List, Any
import json
import os
from datetime import datetime, timedelta

class ConversationAnalytics:
    """Analytics engine for chatbot conversations"""
    
    def __init__(self, conversation_memory):
        self.conversation_memory = conversation_memory
        
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate analytics dashboard data without visualization dependencies"""
        all_conversations = self.conversation_memory.get_all_conversations()
        
        if not all_conversations:
            return {"error": "No conversation data available"}
            
        # Convert to list for easier analysis
        messages = []
        for conv in all_conversations:
            for msg in conv.messages:
                messages.append({
                    "conversation_id": conv.id,
                    "user_id": conv.user_id,
                    "timestamp": msg.timestamp,
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Basic statistics
        total_conversations = len(all_conversations)
        user_ids = set()
        for conv in all_conversations:
            user_ids.add(conv.user_id)
        total_users = len(user_ids)
        total_messages = len(messages)
        avg_messages_per_conversation = total_messages / total_conversations if total_conversations > 0 else 0
        
        # Message content analysis
        user_questions = [msg["content"] for msg in messages if msg["role"] == "user"]
        all_words = " ".join(user_questions).lower().split()
        common_words = Counter(all_words).most_common(20)
        
        # Time analysis
        dates = {}
        for msg in messages:
            try:
                date_str = msg["timestamp"].split("T")[0]  # Extract YYYY-MM-DD
                if date_str in dates:
                    dates[date_str] += 1
                else:
                    dates[date_str] = 1
            except:
                pass
                
        messages_by_date = [{"date": k, "count": v} for k, v in dates.items()]
        
        return {
            "overview": {
                "total_conversations": total_conversations,
                "total_users": total_users,
                "total_messages": total_messages,
                "avg_messages_per_conversation": avg_messages_per_conversation
            },
            "content_analysis": {
                "common_words": common_words
            },
            "time_analysis": {
                "messages_by_date": messages_by_date
            }
        }