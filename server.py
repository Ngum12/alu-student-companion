import os
import sys

# Explicitly disable CUDA before any other imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Print diagnostic information
print("=== RENDER STARTUP: Server initialization ===")
print(f"=== RENDER STARTUP: PORT={os.environ.get('PORT')} ===")
print(f"=== RENDER STARTUP: Python version: {sys.version} ===")

import logging
import threading
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("render-startup")

# Define request models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    options: Optional[Dict[str, Any]] = None

# Create a lightweight app that responds immediately
app = FastAPI(title="ALU Chatbot Backend")

# CORS MUST be configured FIRST - before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global loading state
is_loading = True
main_app_ready = False
has_error = False
error_message = ""

# Basic routes that respond instantly
@app.get("/")
async def root():
    return {
        "status": "ALU Chatbot backend is running",
        "full_app_ready": main_app_ready,
        "is_loading": is_loading,
        "has_error": has_error
    }

@app.get("/health")
async def health():
    global has_error, error_message
    if has_error:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": error_message,
                "is_loading": is_loading,
                "full_app_ready": main_app_ready
            }
        )
    return {"status": "healthy", "full_app_ready": main_app_ready}

@app.post("/api/chat")
async def chat_minimal(request: ChatRequest):
    global main_app_ready, has_error
    
    if has_error:
        return {
            "response": f"I'm sorry, the AI system encountered an error during initialization: {error_message}",
            "conversation_id": "error-1"
        }
    
    if not main_app_ready:
        return {
            "response": f"I'm still loading my knowledge base. Your question about '{request.message}' will be answered when I'm ready.",
            "conversation_id": "loading-1",
            "loading_status": True
        }
    
    # If we get here, the main app is ready - import the function
    try:
        from main import process_chat_internal
        result = process_chat_internal(request)
        return result
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return {
            "response": "I'm sorry, I encountered an error processing your request.",
            "error_type": "processing_error"
        }

# Background loading function with memory management
def load_main_app():
    global is_loading, main_app_ready, has_error, error_message
    
    try:
        # Step 1: Set aggressive memory limits before importing anything
        if sys.platform != "win32":
            try:
                import resource
                # Limit to 1GB memory (Render has 512MB on free tier)
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, hard))
                print("✅ Memory limited to 1GB")
            except:
                print("⚠️ Could not set memory limits")
        
        # Step 2: Import core components individually with careful error handling
        print("🔄 Starting to load components...")
        
        # These are lightweight imports that should work even with tight memory
        try:
            from enhanced_capabilities.conversation_memory import ConversationMemory
            os.makedirs("./data", exist_ok=True)
            conversation_memory = ConversationMemory(persistence_path="./data/conversations.json")
            print("✅ ConversationMemory loaded")
        except Exception as e:
            print(f"⚠️ Failed to load ConversationMemory: {e}")
            raise
        
        # Try to import the remaining components
        try:
            from main import process_chat_internal
            print("✅ Imported process_chat_internal function")
            
            # Set the global flag to indicate that the main app is ready
            main_app_ready = True
            is_loading = False
            print("✅ Main app is now READY")
        except Exception as e:
            print(f"⚠️ Failed to import main app: {e}")
            has_error = True
            error_message = str(e)
            raise
            
    except Exception as e:
        is_loading = False
        has_error = True
        error_message = str(e)
        print(f"❌ CRITICAL ERROR loading components: {e}")

# Start loading in background
bg_thread = threading.Thread(target=load_main_app)
bg_thread.daemon = True
bg_thread.start()

# This runs when executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"⚡ Starting server on port {port} ⚡")
    uvicorn.run(app, host="0.0.0.0", port=port)