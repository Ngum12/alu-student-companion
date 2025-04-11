import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI(title="ALU Chatbot Backend (Minimal)")

# CORS configuration
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", 
                         "http://localhost:3000,http://localhost:3001,https://alu-student-companion.onrender.com").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define minimal request model
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    options: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {"status": "ALU Chatbot backend is running (minimal version)"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "minimal"}

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    return {
        "response": f"This is a minimal response to: '{request.message}'. The full AI chatbot is being initialized.",
        "conversation_id": "minimal-1"
    }

# Start server when run directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)