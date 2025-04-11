
import os
from pathlib import Path
from typing import Dict

# Directory for storing prompt templates
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
PROMPTS_DIR = DATA_DIR / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)

# Default prompt templates for different situations
DEFAULT_PROMPTS = {
    "general": """
As an AI assistant for African Leadership University (ALU) students, I aim to provide helpful, accurate, and contextual responses.

Context information:
{context}

Chat history:
{history}

User query: {query}

Respond to the query using the context provided, and maintain a friendly and supportive tone.
""",
    "academic": """
As an academic assistant for ALU students, I provide educational guidance and help with coursework.

Relevant academic information:
{context}

Previous conversation:
{history}

Student query: {query}

Provide an academically sound response that helps the student learn and understand the topic better.
""",
    "admin": """
As an administrative assistant for ALU staff, I provide detailed information about university processes, policies, and operations.

Relevant administrative information:
{context}

Previous conversation:
{history}

Admin query: {query}

Provide a comprehensive response with specific details, references to relevant policies or procedures, and actionable next steps if applicable.
""",
    "faculty": """
As a faculty assistant for ALU educators, I help with teaching resources, student management, and academic planning.

Relevant faculty information:
{context}

Previous conversation:
{history}

Faculty query: {query}

Provide a detailed response with teaching recommendations, resource suggestions, or administrative guidance as needed.
"""
}

class PromptTemplateManager:
    """Manages loading and accessing prompt templates"""
    
    def __init__(self):
        self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self):
        """Initialize prompt templates"""
        # Save default prompts if they don't exist
        for prompt_type, template in DEFAULT_PROMPTS.items():
            prompt_path = PROMPTS_DIR / f"{prompt_type}_prompt.txt"
            if not prompt_path.exists():
                with open(prompt_path, "w") as f:
                    f.write(template)
    
    def get_prompt_template(self, role: str = "student", query: str = "") -> str:
        """Get the appropriate prompt template based on user role and query"""
        # Map roles to prompt types
        role_to_prompt = {
            "student": "general",
            "admin": "admin",
            "faculty": "faculty"
        }
        
        prompt_type = role_to_prompt.get(role, "general")
        if role == "student" and "academic" in self._query_category(query):
            prompt_type = "academic"
            
        # Load the prompt template
        prompt_path = PROMPTS_DIR / f"{prompt_type}_prompt.txt"
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                return f.read()
        else:
            # Fall back to default template
            return DEFAULT_PROMPTS.get(prompt_type, DEFAULT_PROMPTS["general"])
    
    def _query_category(self, query: str) -> list:
        """Simple categorization of queries"""
        categories = []
        
        # Academic keywords
        academic_keywords = ["course", "assignment", "exam", "study", "learn", 
                           "class", "lecture", "professor", "grade", "academic"]
        
        # Administrative keywords
        admin_keywords = ["register", "enrollment", "tuition", "deadline", "policy", 
                         "form", "application", "schedule", "payment", "administrative"]
        
        # Simple keyword matching
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in academic_keywords):
            categories.append("academic")
            
        if any(keyword in query_lower for keyword in admin_keywords):
            categories.append("administrative")
            
        if not categories:
            categories.append("general")
            
        return categories
