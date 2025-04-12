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

# Function to determine if a message is school-related
def is_school_related(message):
    """Check if a message is related to school/academic topics"""
    school_keywords = [
        "class", "course", "assignment", "homework", "exam", "study", 
        "professor", "lecture", "student", "academic", "university", 
        "college", "ALU", "alu", "deadline", "syllabus", "curriculum"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in school_keywords)

# Cross-platform imports and utilities
if platform.system() != "Windows":
    # Linux-specific imports
    import resource
    
    # Linux memory limit function
    def limit_memory(max_gb=2):
        """Limit memory usage on Linux"""
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Set to max_gb GB or keep hard limit if lower
        max_bytes = max_gb * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (min(max_bytes, hard), hard))
        print(f"✅ Memory limited to {max_gb}GB on Linux")
        
    # Linux timeout handler using SIGALRM
    def timeout_handler(signum, frame):
        """Handle timeouts on Linux"""
        raise TimeoutError("Function call timed out")
else:
    # Windows mock implementations
    def limit_memory(max_gb=2):
        """Mock memory limit function for Windows"""
        print(f"ℹ️ Memory limiting not available on Windows (would be {max_gb}GB on Linux)")
    
    # Windows timeout handler (no SIGALRM)
    def timeout_handler(signum, frame):
        """Mock timeout handler for Windows"""
        pass  # Windows can't use this signal mechanism

# More aggressive memory management for Render
def cleanup_memory():
    """Cross-platform memory cleanup"""
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Linux-specific memory release
    if platform.system() != "Windows":
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            # Return memory to OS
            libc.malloc_trim(0)
            print("✅ Released memory back to OS (Linux)")
        except Exception as e:
            print(f"⚠️ Error releasing memory to OS: {e}")

# Set environment variables to reduce memory usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"

# Force CPU usage and single thread
if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.set_num_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alu-chatbot")

# Pre-define components so we can set them one by one
document_processor = None
retrieval_engine = None
prompt_engine = None
nyptho = None
conversation_memory = None

# Function to handle different types of questions with appropriate capabilities
def handle_question(question, retrieval_func):
    """
    Route questions to appropriate handlers based on content
    Args:
        question: The user's question
        retrieval_func: Function to retrieve context if needed
    
    Returns:
        Dict with answer, source, and additional_info
    """
    # Simple router for demonstration - in production you'd have more sophisticated detection
    if any(term in question.lower() for term in ["calculate", "solve", "equation", "math"]):
        # Mock math solver
        return {
            "answer": "This appears to be a math question. Here's the solution:",
            "source": "math_solver",
            "additional_info": ["Step 1: Identify the equation", "Step 2: Solve for variables"]
        }
    elif any(term in question.lower() for term in ["code", "program", "function", "algorithm"]):
        # Mock code support
        return {
            "answer": "Here's a code example that might help:",
            "source": "code_support",
            "additional_info": {
                "language": "python",
                "code": "def example():\n    return 'Hello World!'"
            }
        }
    elif any(term in question.lower() for term in ["news", "recent", "latest"]):
        # Mock web search
        return {
            "answer": "Based on recent information I found:",
            "source": "web_search",
            "additional_info": {
                "snippets": ["Relevant information from the web"],
                "links": ["https://example.com/resource1", "https://example.com/resource2"]
            }
        }
    else:
        # Use context retrieval as fallback
        context = retrieval_func(question)
        return {
            "answer": "Based on ALU's information: This is a response based on retrieved context.",
            "source": "knowledge_base",
            "additional_info": {"context_used": len(context)}
        }

# Function to lazily initialize components only when needed
def get_component(name):
    global document_processor, retrieval_engine, prompt_engine, nyptho, conversation_memory
    
    if name == "document_processor" and document_processor is None:
        from document_processor import DocumentProcessor
        document_processor = DocumentProcessor()
        
    elif name == "retrieval_engine" and retrieval_engine is None:
        from retrieval_engine_extended import ExtendedRetrievalEngine
        retrieval_engine = ExtendedRetrievalEngine()
        
    elif name == "prompt_engine" and prompt_engine is None:
        from prompt_engine import PromptEngine
        prompt_engine = PromptEngine()
        
    elif name == "nyptho" and nyptho is None:
        from prompt_engine.nyptho_integration import NypthoIntegration
        nyptho = NypthoIntegration()
        
    elif name == "conversation_memory" and conversation_memory is None:
        from enhanced_capabilities.conversation_memory import ConversationMemory
        os.makedirs("./data", exist_ok=True)
        conversation_memory = ConversationMemory(persistence_path="./data/conversations.json")
        conversation_memory.load_from_disk()
    
    if name == "document_processor": return document_processor
    elif name == "retrieval_engine": return retrieval_engine
    elif name == "prompt_engine": return prompt_engine
    elif name == "nyptho": return nyptho
    elif name == "conversation_memory": return conversation_memory

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
# Cross-platform function to execute with timeout
def execute_with_timeout(func, timeout_seconds=30, **kwargs):
    """Execute a function with a timeout on both Windows and Linux platforms"""
    result = None
    error = None
    
    # For Linux, use signal-based timeout
    if platform.system() != "Windows":
        def timeout_handler(signum, frame):
            raise TimeoutError("Function execution timed out")
        
        # Set the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            result = func(**kwargs)
            # Cancel the alarm if successful
            signal.alarm(0)
        except Exception as e:
            error = e
            # Cancel the alarm on exception too
            signal.alarm(0)
    else:
        # For Windows, use threading-based timeout
        def worker():
            nonlocal result, error
            try:
                result = func(**kwargs)
            except Exception as e:
                error = e
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            error = TimeoutError("Function execution timed out")
    
    return result, error

def process_chat_internal(request: ChatRequest):
    """Internal function that can be imported by server.py"""
    user_message = request.message
    user_id = request.options.get("user_id", "anonymous") if request.options else "anonymous"
    conversation_id = request.options.get("conversation_id") if request.options else None

@app.get("/health")
async def health():
    """Health check endpoint with detailed system status"""
    try:
        # Basic health check - will succeed even if other components fail
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "document_processor": get_component("document_processor") is not None,
                "retrieval_engine": get_component("retrieval_engine") is not None,
                "prompt_engine": get_component("prompt_engine") is not None,
                "conversation_memory": get_component("conversation_memory") is not None
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/debug")
async def debug_info():
    """Debug endpoint with detailed system info"""
    import sys
    import psutil
    process = psutil.Process()
    
    return {
        "status": "running",
        "python_version": sys.version,
        "platform": platform.system(),
        "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
        "components_loaded": {
            "document_processor": get_component("document_processor") is not None,
            "retrieval_engine": get_component("retrieval_engine") is not None,
            "prompt_engine": get_component("prompt_engine") is not None,
            "conversation_memory": get_component("conversation_memory") is not None
        }
    }

def process_chat_internal(request: ChatRequest):
    """Internal function that can be imported by server.py"""
    user_message = request.message
    user_id = request.options.get("user_id", "anonymous") if request.options else "anonymous"
    conversation_id = request.options.get("conversation_id") if request.options else None
    
    print(f"Processing message: '{user_message}'")
    print(f"Is school related: {is_school_related(user_message)}")
    
    try:
        # Add user message to conversation memory
        conversation_memory = get_component("conversation_memory")
        conversation_memory.add_message(user_id, "user", user_message, conversation_id)
        
        # Get conversation history for context
        conversation = conversation_memory.get_active_conversation(user_id)
        conversation_history = conversation.get_formatted_history()
        
        # First, check if this is a specialized query that should use enhanced capabilities
        if not is_school_related(user_message):
            try:
                print(f"Trying enhanced capabilities for: '{user_message}'")
                
                # Use the enhanced capabilities router
                retrieval_engine = get_component("retrieval_engine")
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
        retrieval_engine = get_component("retrieval_engine")
        context_docs = retrieval_engine.retrieve_context(
            query=user_message,
            role="student"  # Default role
        )
        
        # Generate response using the prompt engine
        prompt_engine = get_component("prompt_engine")
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

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    try:
        # Use cross-platform timeout for processing
        result, error = execute_with_timeout(
            process_chat_internal,
            timeout_seconds=30,
            request=request
        )
        
        if error:
            if isinstance(error, TimeoutError):
                return {
                    "response": "I'm sorry, processing your request took too long. Please try a simpler question.",
                    "error_type": "timeout"
                }
            return {
                "response": "I'm sorry, I encountered an error processing your request.",
                "error_type": "processing_error"
            }
            
        return result
    except Exception as e:
        logger.error(f"Error in process_chat: {str(e)}")
        return {
            "response": "I'm sorry, I encountered an unexpected error.",
            "error_type": "general_error"
        }