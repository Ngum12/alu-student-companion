print("=== STARTUP: Beginning application initialization ===")
import os
print(f"=== STARTUP: PORT environment variable: {os.environ.get('PORT')} ===")

import json
import random
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import gc
import torch
import time
import sys
import logging
import platform
import signal
import threading

# Set environment variables to reduce memory usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"

# Force CPU usage and single thread
if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.set_num_threads(1)

# Add a memory cleanup function
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alu-chatbot")

# Import the modules
from document_processor import DocumentProcessor
from retrieval_engine_extended import ExtendedRetrievalEngine
from prompt_engine import PromptEngine
from prompt_engine.nyptho_integration import NypthoIntegration
from enhanced_capabilities.capability_router import handle_question, is_school_related
from enhanced_capabilities.conversation_memory import ConversationMemory

# Create FastAPI app
app = FastAPI(title="ALU Chatbot Backend")

# Get CORS settings from environment or use default
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,https://alu-student-companion.onrender.com").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace the existing component initialization code with this
try:
    # Initialize components one by one with explicit error handling
    try:
        document_processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized")
    except Exception as e:
        print(f"⚠️ DocumentProcessor init failed: {e}")
        document_processor = None
        
    try:
        retrieval_engine = ExtendedRetrievalEngine()
        print("✅ ExtendedRetrievalEngine initialized")
    except Exception as e:
        print(f"⚠️ ExtendedRetrievalEngine init failed: {e}")
        retrieval_engine = None
        
    try:
        prompt_engine = PromptEngine()
        print("✅ PromptEngine initialized")
    except Exception as e:
        print(f"⚠️ PromptEngine init failed: {e}")
        prompt_engine = None
        
    try:
        nyptho = NypthoIntegration()
        print("✅ NypthoIntegration initialized")
    except Exception as e:
        print(f"⚠️ NypthoIntegration init failed: {e}")
        nyptho = None
        
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    try:
        conversation_memory = ConversationMemory(persistence_path="./data/conversations.json")
        conversation_memory.load_from_disk()
        print("✅ ConversationMemory initialized and loaded")
    except Exception as e:
        print(f"⚠️ ConversationMemory init failed: {e}")
        conversation_memory = None
        
except Exception as e:
    print(f"⚠️ CRITICAL INIT ERROR: {e}")
    # Don't exit - provide minimal functionality instead

# Define request models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    options: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    role: str = "student"
    conversation_history: List[Dict[str, Any]] = []
    options: Optional[Dict[str, Any]] = None

class DocumentMetadata(BaseModel):
    title: str
    source: str
    date: Optional[str] = None

class PersonalitySettings(BaseModel):
    helpfulness: float
    creativity: float
    precision: float
    friendliness: float

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ALU Chatbot backend is running"}

@app.get("/health")
async def health():
    """Health check endpoint with detailed system status"""
    try:
        # Basic health check - will succeed even if other components fail
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "document_processor": document_processor is not None,
                "retrieval_engine": retrieval_engine is not None,
                "prompt_engine": prompt_engine is not None,
                "conversation_memory": conversation_memory is not None
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    user_message = request.message
    user_id = request.options.get("user_id", "anonymous") if request.options else "anonymous"
    conversation_id = request.options.get("conversation_id") if request.options else None
    
    print(f"Processing message: '{user_message}'")
    print(f"Is school related: {is_school_related(user_message)}")
    
    try:
        # Add user message to conversation memory
        conversation_memory.add_message(user_id, "user", user_message, conversation_id)
        
        # Get conversation history for context
        conversation = conversation_memory.get_active_conversation(user_id)
        conversation_history = conversation.get_formatted_history()
        
        # First, check if this is a specialized query that should use enhanced capabilities
        if not is_school_related(user_message):
            try:
                print(f"Trying enhanced capabilities for: '{user_message}'")
                
                # Use the enhanced capabilities router
                result = handle_question(
                    user_message, 
                    lambda q: retrieval_engine.retrieve_context(q)
                )
                
                print(f"Selected capability: {result['source']}")
                
                # Format the response based on which capability handled it
                if result["source"] == "math_solver":
                    steps = "\n".join(result["additional_info"]) if result["additional_info"] else ""
                    response = f"{result['answer']}\n\n{steps}"
                elif result["source"] == "web_search":
                    snippets = result["additional_info"]["snippets"][:2] if "snippets" in result["additional_info"] else []
                    links = result["additional_info"]["links"][:2] if "links" in result["additional_info"] else []
                    
                    sources = ""
                    for i, link in enumerate(links):
                        sources += f"\n- [{link}]({link})"
                    
                    response = f"{result['answer']}\n\nSources:{sources}"
                elif result["source"] == "code_support":
                    # Properly format code responses
                    code_result = result.get("additional_info", {})
                    language = code_result.get("language", "text")
                    code = code_result.get("code", "")
                    
                    # Format with proper code blocks
                    response = f"{result['answer']}\n\n```{language}\n{code}\n```"
                else:
                    response = result["answer"]
                
                # Add AI response to conversation memory
                conversation_memory.add_message(user_id, "assistant", response, conversation.id)
                
                # Periodically save conversations to disk
                if random.random() < 0.1:  # 10% chance to save after each message
                    conversation_memory.save_to_disk()
                
                cleanup_memory()
                return {
                    "response": response,
                    "conversation_id": conversation.id
                }
            except Exception as e:
                print(f"Enhanced capabilities error: {e}")
                # Continue to existing document retrieval code
        
        # Get relevant context from the retrieval engine
        context_docs = retrieval_engine.retrieve_context(
            query=user_message,
            role="student"  # Default role
        )
        
        # Generate response using the prompt engine
        response = prompt_engine.generate_response(
            query=user_message,
            context=context_docs,
            conversation_history=conversation_history,
            role="student",
            options={}
        )
        
        # Add AI response to conversation memory
        conversation_memory.add_message(user_id, "assistant", response, conversation.id)
        
        # Extract sources for attribution
        sources = []
        for doc in context_docs[:3]:  # Top 3 sources
            if doc.metadata and 'source' in doc.metadata:
                source = {
                    'title': doc.metadata.get('title', 'ALU Knowledge Base'),
                    'source': doc.metadata.get('source', 'ALU Brain')
                }
                if source not in sources:  # Avoid duplicates
                    sources.append(source)
        
        cleanup_memory()
        return {
            "response": response,
            "sources": sources,
            "engine": "alu_prompt_engine",
            "conversation_id": conversation.id
        }
        
    except Exception as e:
        print(f"Error processing chat: {e}")
        cleanup_memory()
        return {"response": "I'm sorry, I couldn't process your request due to a technical error."}