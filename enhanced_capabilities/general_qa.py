from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import os
from typing import Dict, Any

# Initialize the QA pipeline (lazy loading to save memory)
_qa_pipeline = None

def get_qa_pipeline():
    """Initialize and return the QA pipeline."""
    global _qa_pipeline
    
    if (_qa_pipeline is None):
        # Use a small model that can run locally
        model_name = "distilbert-base-cased-distilled-squad"
        
        try:
            _qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name
            )
        except Exception as e:
            print(f"Error loading QA model: {e}")
            return None
    
    return _qa_pipeline

def ask_transformer_model(question: str, context: str = None) -> str:
    """
    Simplified version that doesn't use transformers.
    This is a placeholder implementation that returns a fixed response.
    
    Args:
        question: The question to answer
        context: Optional context to help answer the question
        
    Returns:
        String containing the answer
    """
    return "I don't have enough information to answer that question locally. Let me search the web for you."