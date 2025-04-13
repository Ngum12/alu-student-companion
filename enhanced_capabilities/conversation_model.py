import os
import time
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class ConversationModelHandler:
    """Handles conversational capabilities using a Hugging Face language model"""
    
    def __init__(self, model_name="distilgpt2"):
        """Initialize with a small but effective language model"""
        self.model_name = model_name
        
        print(f"Loading conversational model: {model_name}")
        start_time = time.time()
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Simplified model loading without problematic parameters
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            load_time = time.time() - start_time
            print(f"✅ Conversational model loaded in {load_time:.2f} seconds")
            self.model_loaded = True
        except Exception as e:
            print(f"⚠️ Error loading conversational model: {e}")
            self.model_loaded = False
    
    def format_conversation_prompt(self, message: str, conversation_history: List[Dict[str, str]], 
                                   max_history: int = 3) -> str:
        """Format the conversation history into a prompt for the model"""
        prompt = "The following is a conversation with a helpful AI assistant for African Leadership University students.\n\n"
        
        # Add up to max_history previous exchanges for context
        history_items = conversation_history[-max_history*2:] if conversation_history else []
        
        for item in history_items:
            if "user" in item:
                prompt += f"Human: {item['user']}\n"
            if "assistant" in item:
                prompt += f"AI: {item['assistant']}\n"
                
        # Add the current message
        prompt += f"Human: {message}\nAI:"
        
        return prompt
    
    def get_response(self, message: str, conversation_history: List[Dict[str, str]]) -> str:
        """Generate a conversational response using the model"""
        if not self.model_loaded:
            return "I'm here to help with your questions about ALU. What would you like to know?"
        
        try:
            # Format the prompt with conversation history
            prompt = self.format_conversation_prompt(message, conversation_history)
            
            # Generate response with the model - ADD padding=True HERE
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "AI:" in response:
                response = response.split("AI:")[-1].strip()
                
            # Ensure response isn't empty
            if not response or len(response) < 10:
                return "I'm here to help with your questions about ALU. What would you like to know?"
                
            return response
        except Exception as e:
            print(f"Error generating model response: {e}")
            return "I'm here to help with your questions about ALU. What would you like to know?"