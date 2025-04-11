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
import resource
import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alu-chatbot")

# Remove Hugging Face specific code
# Instead, directly use lightweight settings for all deployments
torch.set_grad_enabled(False)  # Disable gradient computation
device = "cpu"  # Force CPU usage

# Set environment variables to reduce memory usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["TRANSFORMERS_CACHE_MAX_SIZE"] = "2G"  # 2GB max cache

# Force CPU usage and single thread
if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.set_num_threads(1)

# Add a memory cleanup function
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Set memory limit (2GB)
def limit_memory():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, hard))

# Set function timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

# Use like this in heavy functions:
# signal.signal(signal.SIGALRM, timeout_handler)
# signal.alarm(30)  # 30 second timeout
# try:
#     heavy_function()
# except TimeoutError:
#     logger.error("Function timed out")
# finally:
#     signal.alarm(0)  # Cancel alarm

# Import the modules
from document_processor import DocumentProcessor
from retrieval_engine_extended import ExtendedRetrievalEngine
from prompt_engine import PromptEngine
from prompt_engine.nyptho_integration import NypthoIntegration
from enhanced_capabilities.capability_router import handle_question, is_school_related
from enhanced_capabilities.conversation_memory import ConversationMemory

# Add this near the top of your file
_document_processor = None
_retrieval_engine = None
_prompt_engine = None
_nyptho = None

# Function to get components only when needed
def get_document_processor():
    global _document_processor
    if _document_processor is None:
        try:
            _document_processor = DocumentProcessor()
            logger.info("✅ DocumentProcessor initialized on first use")
        except Exception as e:
            logger.warning(f"⚠️ DocumentProcessor init failed: {e}")
            _document_processor = None
    return _document_processor

# Similar functions for other components
def get_retrieval_engine():
    global _retrieval_engine
    if _retrieval_engine is None:
        try:
            _retrieval_engine = ExtendedRetrievalEngine()
            logger.info("✅ ExtendedRetrievalEngine initialized on first use")
        except Exception as e:
            logger.warning(f"⚠️ ExtendedRetrievalEngine init failed: {e}")
            _retrieval_engine = None
    return _retrieval_engine

def get_prompt_engine():
    global _prompt_engine
    if _prompt_engine is None:
        try:
            _prompt_engine = PromptEngine()
            logger.info("✅ PromptEngine initialized on first use")
        except Exception as e:
            logger.warning(f"⚠️ PromptEngine init failed: {e}")
            _prompt_engine = None
    return _prompt_engine

def get_nyptho():
    global _nyptho
    if _nyptho is None:
        try:
            _nyptho = NypthoIntegration()
            logger.info("✅ NypthoIntegration initialized on first use")
        except Exception as e:
            logger.warning(f"⚠️ NypthoIntegration init failed: {e}")
            _nyptho = None
    return _nyptho

# Create FastAPI app
app = FastAPI(
    title="ALU Chatbot Backend",
    root_path=os.environ.get("ROOT_PATH", "")  # Add this line
)

# Properly configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://alu-student-companion.onrender.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize conversation memory
try:
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    conversation_memory = ConversationMemory(persistence_path="./data/conversations.json")
    conversation_memory.load_from_disk()
    logger.info("✅ ConversationMemory initialized and loaded")
except Exception as e:
    logger.warning(f"⚠️ ConversationMemory init failed: {e}")
    conversation_memory = None

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
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint that doesn't use ML components"""
    return {
        "status": "ok",
        "message": "This is a test response that doesn't use ML components",
        "time": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    try:
        # Memory intensive operations
        # Set a short timeout for initialization
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout for initialization
        
        try:
            # Use lazy loading
            processor = get_document_processor()
            engine = get_retrieval_engine()
            
            # Cancel the initialization timeout
            signal.alarm(0)
            
            # Rest of your existing code...
            
        except TimeoutError:
            logger.error("Component initialization timed out")
            return {
                "response": "I'm currently experiencing high demand. Please try a simpler question or try again later.",
                "error_type": "timeout"
            }
        finally:
            signal.alarm(0)  # Always cancel alarm
            
    except Exception as e:
        logger.error(f"Critical error in process_chat: {str(e)}")
        return {
            "response": "I'm sorry, I couldn't process your request at this time. Please try again later.",
            "error_type": "general_error"
        }
    finally:
        cleanup_memory()

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    # For long requests, use background tasks
    try:
        if request.async_mode:
            # Process in background
            background_tasks.add_task(process_chat_request, request)
            return {"status": "processing", "message": "Your request is being processed"}
        else:
            # Process synchronously with timeout
            return await asyncio.wait_for(process_chat_request(request), timeout=25.0)
    except asyncio.TimeoutError:
        return {"error": "Request timed out, please try again"}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": "An error occurred"}

@app.post("/generate")
async def generate_response(request: QueryRequest):
    """Generate a response for the user query"""
    try:
        # Get relevant context from the retrieval engine
        context_docs = get_retrieval_engine().retrieve_context(
            query=request.query, 
            role=request.role
        )
        
        # Check if we should use Nyptho
        use_nyptho = False
        if request.options and "use_nyptho" in request.options:
            use_nyptho = request.options["use_nyptho"]
        
        # Set model ID for observation
        model_id = "standard_engine"
        
        # Generate response using appropriate engine
        if use_nyptho and get_nyptho().get_status()["ready"]:
            # Use Nyptho for response
            personality = None
            if request.options and "personality" in request.options:
                personality = request.options["personality"]
                
            response = get_nyptho().generate_response(
                query=request.query,
                context=context_docs,
                personality=personality
            )
            model_id = "nyptho"
        else:
            # Use standard prompt engine
            response = get_prompt_engine().generate_response(
                query=request.query,
                context=context_docs,
                conversation_history=request.conversation_history,
                role=request.role,
                options=request.options
            )
        
        # Have Nyptho observe this interaction (it learns from all responses)
        if model_id != "nyptho":  # Don't observe itself
            get_nyptho().observe_model(
                query=request.query,
                response=response,
                model_id=model_id,
                context=context_docs
            )
        
        return {
            "response": response,
            "sources": [doc.metadata for doc in context_docs[:3]] if context_docs else [],
            "engine": model_id
        }
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(None),
    source: str = Form("user-upload"),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a document into the vector store"""
    try:
        # Process the document
        doc_id = await get_document_processor().process_document(file, title, source)
        
        # Add background task to update the vector store
        if background_tasks:
            background_tasks.add_task(
                get_retrieval_engine().update_vector_store,
                doc_id
            )
        
        return {"status": "success", "message": "Document uploaded successfully", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all available documents in the knowledge base"""
    try:
        documents = get_document_processor().list_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base"""
    try:
        success = get_document_processor().delete_document(doc_id)
        if success:
            # Update the vector store to remove the document's embeddings
            get_retrieval_engine().remove_document(doc_id)
            return {"status": "success", "message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild the vector index with all documents"""
    try:
        background_tasks.add_task(get_retrieval_engine().rebuild_index)
        return {"status": "success", "message": "Index rebuild started in the background"}
    except Exception as e:
        logger.error(f"Error starting index rebuild: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Nyptho-specific endpoints
@app.get("/nyptho/status")
async def get_nyptho_status():
    """Get the current status of Nyptho"""
    try:
        status = get_nyptho().get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting Nyptho status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nyptho/personality")
async def set_nyptho_personality(settings: PersonalitySettings):
    """Update Nyptho's personality settings"""
    try:
        result = get_nyptho().set_personality({
            "helpfulness": settings.helpfulness,
            "creativity": settings.creativity,
            "precision": settings.precision,
            "friendliness": settings.friendliness
        })
        return result
    except Exception as e:
        logger.error(f"Error updating personality: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-stats")
async def get_search_stats():
    """Get search engine performance statistics"""
    try:
        search_stats = get_retrieval_engine().alu_brain.search_engine.get_search_stats()
        return search_stats
    except Exception as e:
        logger.error(f"Error getting search stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    """Handler for application shutdown"""
    logger.info("Saving conversation memory...")
    conversation_memory.save_to_disk()
    logger.info("Shutting down Nyptho...")
    try:
        get_nyptho().shutdown()
    except:
        pass  # Ignore errors during shutdown