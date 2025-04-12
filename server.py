import os
import sys
import logging
import time
import threading
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("render-startup")

# Print diagnostic information
print("=== RENDER STARTUP: Server initialization ===")
print(f"=== RENDER STARTUP: PORT={os.environ.get('PORT')} ===")
print(f"=== RENDER STARTUP: Python version: {sys.version} ===")

# Define request models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    options: Optional[Dict[str, Any]] = None

# Create a single shared app
app = FastAPI(title="ALU Chatbot Backend")

# Add CORS middleware
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", 
                      "http://localhost:3000,http://localhost:3001,https://alu-student-companion.onrender.com").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global flag and components
full_app_ready = False
full_chat_processor = None

# Basic routes
@app.get("/")
async def root():
    return {"status": "ALU Chatbot backend is running", "full_app_ready": full_app_ready}

@app.get("/health")
async def health():
    return {"status": "healthy", "full_app_ready": full_app_ready}

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    global full_app_ready, full_chat_processor
    
    if not full_app_ready or full_chat_processor is None:
        return {
            "response": f"This is a minimal response to: '{request.message}'. The full AI chatbot is being initialized.",
            "conversation_id": "minimal-1"
        }
    
    # Use the full functionality once loaded
    try:
        return full_chat_processor(request)
    except Exception as e:
        print(f"Error in full chat processing: {e}")
        return {
            "response": f"Error processing your request: {str(e)}",
            "error_type": "processing_error"
        }

# Function to load components in background
def load_full_components():
    global full_app_ready, full_chat_processor
    
    try:
        print("🔄 Starting to load full app components...")
        start_time = time.time()
        
        # Import the full app's processing function
        from main import process_chat_internal
        full_chat_processor = process_chat_internal
        
        # Mark as ready
        full_app_ready = True
        
        elapsed = time.time() - start_time
        print(f"✅ Full components loaded successfully in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"⚠️ Critical error loading components: {e}")

# Start loading components in background
bg_thread = threading.Thread(target=load_full_components)
bg_thread.daemon = True
bg_thread.start()

# Start server when run directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"⚡ Starting server on port {port} ⚡")
    uvicorn.run(app, host="0.0.0.0", port=port)