
from typing import List, Dict, Any, Optional

from retrieval_engine import Document
from .templates import PromptTemplateManager
from .response_generator import ResponseGenerator
from .formatters import ContentFormatter

class PromptEngine:
    """
    Handles prompt engineering and response generation:
    - Dynamic prompt construction based on query, context, and user role
    - Integration with LLM services
    - Response formatting and enhancement
    """

    def __init__(self):
        self.template_manager = PromptTemplateManager()
        self.response_generator = ResponseGenerator()
        self.formatter = ContentFormatter()
        self.current_query = ""
        print("Prompt Engine initialized with template manager and response generator")
    
    def generate_response(
        self, 
        query: str, 
        context: List[Document], 
        conversation_history: List[Dict[str, Any]] = [],
        role: str = "student",
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response using prompt engineering and context:
        1. Select and fill appropriate prompt template
        2. Call LLM service
        3. Format and return response
        """
        self.current_query = query  # Store for categorization
        
        # Get the appropriate prompt template
        prompt_template = self.template_manager.get_prompt_template(role, query)
        
        # Format context and history
        formatted_context = self.formatter.format_context(context)
        formatted_history = self.formatter.format_conversation_history(conversation_history)
        
        # Fill in the prompt template
        prompt = prompt_template.format(
            query=query,
            context=formatted_context,
            history=formatted_history
        )
        
        # Generate response
        response = self.response_generator.generate_response(query, context, role)
        
        return response
