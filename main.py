print("=== STARTUP: Beginning application initialization ===")
import os
print(f"=== STARTUP: PORT environment variable: {os.environ.get('PORT')} ===")

# Import standard libraries first
import sys
import torch
import threading
import platform
import logging
import psutil
import random
import time
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app before any other imports
app = FastAPI(title="ALU Chatbot Backend")

# Set up CORS
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,https://alu-student-companion.onrender.com").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rest of your imports...
import uvicorn
from fastapi import UploadFile, File, Form, HTTPException, Body, BackgroundTasks
from datetime import datetime
import gc
import signal
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel

# Add this after your other imports
from enhanced_capabilities.conversation_model import ConversationModelHandler

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

# Import the modules
from document_processor import DocumentProcessor
from retrieval_engine_extended import ExtendedRetrievalEngine
from prompt_engine import PromptEngine
from prompt_engine.nyptho_integration import NypthoIntegration
from enhanced_capabilities.capability_router import handle_question, is_school_related
from enhanced_capabilities.conversation_memory import ConversationMemory
# Comment these until the modules are created
from enhanced_capabilities.reasoning_chain import ReasoningChain
from data_integration.alu_api_connector import ALUDataConnector
from analytics.conversation_analytics import ConversationAnalytics

# Timeout execution utility
def execute_with_timeout(func, timeout_seconds=30, *args, **kwargs):
    """
    Run a function with a timeout, works on both Windows and Linux
    Returns tuple (result, error)
    """
    result = [None]
    exception = [None]
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        return None, TimeoutError(f"Function timed out after {timeout_seconds} seconds")
    if exception[0]:
        return None, exception[0]
    return result[0], None

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

    # Initialize advanced capabilities with fallbacks
    try:
        reasoning_chain = ReasoningChain(retrieval_engine, prompt_engine)
        alu_connector = ALUDataConnector()
        print("✅ Advanced capabilities initialized")
    except Exception as e:
        print(f"⚠️ Error initializing advanced capabilities: {e}")
        # Define fallbacks so the code doesn't break if initialization fails
        reasoning_chain = None
        alu_connector = None

    try:
        analytics = ConversationAnalytics(conversation_memory)
        print("✅ Advanced capabilities initialized")
    except Exception as e:
        print(f"⚠️ Error initializing advanced capabilities: {e}")

    # Add this after other component initializations
    try:
        conversation_handler = ConversationModelHandler()
        print("✅ Conversation model handler initialized")
    except Exception as e:
        print(f"⚠️ Conversation model initialization failed: {e}")
        conversation_handler = None
        
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

@app.get("/debug")
async def debug_info():
    """Debug endpoint with detailed system info"""
    process = psutil.Process()
    
    return {
        "status": "running",
        "python_version": sys.version,
        "platform": platform.system(),
        "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
        "components_loaded": {
            "document_processor": document_processor is not None,
            "retrieval_engine": retrieval_engine is not None,
            "prompt_engine": prompt_engine is not None,
            "conversation_memory": conversation_memory is not None
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
                
                # If response came from document search but was very short or low confidence
                if result["source"] == "document_search" and (len(result["answer"]) < 50 or "confidence" in result and result["confidence"] < 0.7):
                    # Try using conversation model instead for common phrases
                    if conversation_handler:
                        try:
                            conv_response = conversation_handler.get_response(user_message, conversation.get_formatted_history())
                            if len(conv_response) > len(result["answer"]):
                                result["answer"] = conv_response
                                result["source"] = "conversation"
                        except Exception as e:
                            print(f"Error using conversation fallback: {e}")
                
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
                    # Replace code generation with friendly explanation
                    response = "I'm sorry, I'm focused on helping with ALU-related questions and can't provide code examples. Could I help you with something about your studies instead?"
                else:
                    response = result["answer"]
                
                # Add conversational fallback for generic or uncertain responses
                if "I don't have specific information" in response or "I don't know" in response or len(response) < 50:
                    if conversation_handler:
                        try:
                            # Get conversation history for context
                            conv_history = conversation.get_formatted_history()
                            conversational_response = conversation_handler.get_response(user_message, conv_history)
                            
                            # If the original response is very short, just use the conversational response
                            if len(response) < 50:
                                response = conversational_response
                            else:
                                # Otherwise append it as a friendly addition
                                response = f"{response}\n\nIs there something else I can help you with about ALU?"
                        except Exception as e:
                            print(f"Error in conversational fallback: {e}")
                            # Continue with original response if conversational fallback fails
                
                # Add a friendly sign-off to responses if they don't already have one
                friendly_endings = ["Feel free to ask", "Let me know", "Hope that helps", "Anything else", "Can I help"]
                if not any(ending in response for ending in friendly_endings) and len(response) > 100:
                    response += "\n\n💡 Is there anything else about ALU you'd like to know?"
                
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
        
        # Add conversational fallback for generic or uncertain responses
        if "I don't have specific information" in response or "I don't know" in response or len(response) < 50:
            if conversation_handler:
                try:
                    # Get conversation history for context
                    conv_history = conversation.get_formatted_history()
                    conversational_response = conversation_handler.get_response(user_message, conv_history)
                    
                    # If the original response is very short, just use the conversational response
                    if len(response) < 50:
                        response = conversational_response
                    else:
                        # Otherwise append it as a friendly addition
                        response = f"{response}\n\nIs there something else I can help you with about ALU?"
                except Exception as e:
                    print(f"Error in conversational fallback: {e}")
                    # Continue with original response if conversational fallback fails
        
        # Add a friendly sign-off to responses if they don't already have one
        friendly_endings = ["Feel free to ask", "Let me know", "Hope that helps", "Anything else", "Can I help"]
        if not any(ending in response for ending in friendly_endings) and len(response) > 100:
            response += "\n\n💡 Is there anything else about ALU you'd like to know?"
        
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

@app.post("/api/image-query")
async def process_image_query(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    """Process queries about uploaded images (documents, schedules, etc.)"""
    try:
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # Load CLIP model
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        
        # Process image and question
        inputs = processor(
            text=[question], 
            images=img, 
            return_tensors="pt", 
            padding=True
        )
        outputs = model(**inputs)
        
        # Calculate similarity between image and text
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        
        # Generate response based on image content
        context_docs = retrieval_engine.retrieve_context(question)
        response = prompt_engine.generate_response(
            query=f"Image question: {question}",
            context=context_docs,
            additional_context={
                "image_text_similarity": float(similarity[0]),
                "confidence": "high" if similarity[0] > 0.25 else "low"
            }
        )
        
        return {"response": response}
    except Exception as e:
        print(f"Error processing image query: {e}")
        return {"response": "I'm sorry, I couldn't process the image. Please try a different image or question."}

@app.post("/api/deep-query")
async def process_deep_query(request: ChatRequest):
    """Process complex queries using multi-step reasoning"""
    try:
        if reasoning_chain is None:
            return {
                "response": "I'm sorry, the advanced reasoning feature is currently unavailable.",
                "error": "capability_unavailable"
            }
            
        result = reasoning_chain.solve(request.message)
        return {
            "response": result["answer"],
            "reasoning_steps": result["reasoning_chain"],
            "steps_taken": result["steps_taken"]
        }
    except Exception as e:
        print(f"Error in deep query: {e}")
        return {
            "response": "I'm sorry, I encountered an error when processing this complex question.",
            "error": str(e)
        }

@app.get("/api/alu-events")
async def get_alu_events(campus: str = "all", days: int = 7):
    """Get upcoming events at ALU"""
    try:
        if alu_connector is None:
            return {
                "events": [],
                "error": "Data connector unavailable"
            }
            
        events = alu_connector.get_upcoming_events(campus, days)
        return {"events": events}
    except Exception as e:
        print(f"Error fetching ALU events: {e}")
        return {"events": [], "error": str(e)}

@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard():
    """Get analytics dashboard data"""
    try:
        if conversation_memory is None or analytics is None:
            return {"error": "Analytics service not available"}
            
        dashboard_data = analytics.generate_dashboard_data()
        return dashboard_data
    except Exception as e:
        print(f"Error generating analytics dashboard: {e}")
        return {"error": f"Could not generate analytics: {str(e)}"}