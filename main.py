import os
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

# Import the modules
from document_processor import DocumentProcessor
from retrieval_engine_extended import ExtendedRetrievalEngine
from prompt_engine import PromptEngine
from prompt_engine.nyptho_integration import NypthoIntegration
from enhanced_capabilities.capability_router import handle_question, is_school_related
from enhanced_capabilities.conversation_memory import ConversationMemory

# Create FastAPI app
app = FastAPI(
    title="ALU Chatbot Backend",
    root_path=os.environ.get("ROOT_PATH", "")  # Add this line
)

# Get CORS settings from environment or use default
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

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
        logger.info("✅ DocumentProcessor initialized")
    except Exception as e:
        logger.warning(f"⚠️ DocumentProcessor init failed: {e}")
        document_processor = None
        
    try:
        retrieval_engine = ExtendedRetrievalEngine()
        logger.info("✅ ExtendedRetrievalEngine initialized")
    except Exception as e:
        logger.warning(f"⚠️ ExtendedRetrievalEngine init failed: {e}")
        retrieval_engine = None
        
    try:
        prompt_engine = PromptEngine()
        logger.info("✅ PromptEngine initialized")
    except Exception as e:
        logger.warning(f"⚠️ PromptEngine init failed: {e}")
        prompt_engine = None
        
    try:
        nyptho = NypthoIntegration()
        logger.info("✅ NypthoIntegration initialized")
    except Exception as e:
        logger.warning(f"⚠️ NypthoIntegration init failed: {e}")
        nyptho = None
        
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    try:
        conversation_memory = ConversationMemory(persistence_path="./data/conversations.json")
        conversation_memory.load_from_disk()
        logger.info("✅ ConversationMemory initialized and loaded")
    except Exception as e:
        logger.warning(f"⚠️ ConversationMemory init failed: {e}")
        conversation_memory = None
        
except Exception as e:
    logger.error(f"⚠️ CRITICAL INIT ERROR: {e}")
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

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    user_message = request.message
    user_id = request.options.get("user_id", "anonymous") if request.options else "anonymous"
    conversation_id = request.options.get("conversation_id") if request.options else None
    
    logger.info(f"Processing message: '{user_message}'")
    logger.info(f"Is school related: {is_school_related(user_message)}")
    
    try:
        # Add user message to conversation memory
        conversation_memory.add_message(user_id, "user", user_message, conversation_id)
        
        # Get conversation history for context
        conversation = conversation_memory.get_active_conversation(user_id)
        conversation_history = conversation.get_formatted_history()
        
        # First, check if this is a specialized query that should use enhanced capabilities
        if not is_school_related(user_message):
            try:
                logger.info(f"Trying enhanced capabilities for: '{user_message}'")
                
                # Use the enhanced capabilities router
                result = handle_question(
                    user_message,
                    search_school_docs_func=lambda q: retrieval_engine.retrieve_context(q),
                    conversation_history=conversation_history
                )
                
                logger.info(f"Selected capability: {result['source']}")
                
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
                logger.error(f"Enhanced capabilities error: {e}")
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
        logger.error(f"Error processing chat: {e}")
        cleanup_memory()
        return {"response": "I'm sorry, I couldn't process your request due to a technical error."}

@app.post("/generate")
async def generate_response(request: QueryRequest):
    """Generate a response for the user query"""
    try:
        # Get relevant context from the retrieval engine
        context_docs = retrieval_engine.retrieve_context(
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
        if use_nyptho and nyptho.get_status()["ready"]:
            # Use Nyptho for response
            personality = None
            if request.options and "personality" in request.options:
                personality = request.options["personality"]
                
            response = nyptho.generate_response(
                query=request.query,
                context=context_docs,
                personality=personality
            )
            model_id = "nyptho"
        else:
            # Use standard prompt engine
            response = prompt_engine.generate_response(
                query=request.query,
                context=context_docs,
                conversation_history=request.conversation_history,
                role=request.role,
                options=request.options
            )
        
        # Have Nyptho observe this interaction (it learns from all responses)
        if model_id != "nyptho":  # Don't observe itself
            nyptho.observe_model(
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
        doc_id = await document_processor.process_document(file, title, source)
        
        # Add background task to update the vector store
        if background_tasks:
            background_tasks.add_task(
                retrieval_engine.update_vector_store,
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
        documents = document_processor.list_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base"""
    try:
        success = document_processor.delete_document(doc_id)
        if success:
            # Update the vector store to remove the document's embeddings
            retrieval_engine.remove_document(doc_id)
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
        background_tasks.add_task(retrieval_engine.rebuild_index)
        return {"status": "success", "message": "Index rebuild started in the background"}
    except Exception as e:
        logger.error(f"Error starting index rebuild: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Nyptho-specific endpoints
@app.get("/nyptho/status")
async def get_nyptho_status():
    """Get the current status of Nyptho"""
    try:
        status = nyptho.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting Nyptho status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nyptho/personality")
async def set_nyptho_personality(settings: PersonalitySettings):
    """Update Nyptho's personality settings"""
    try:
        result = nyptho.set_personality({
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
        search_stats = retrieval_engine.alu_brain.search_engine.get_search_stats()
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
        nyptho.shutdown()
    except:
        pass  # Ignore errors during shutdown

# Run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's provided PORT
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)