import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("render-startup")

# Print diagnostic information
print("=== RENDER STARTUP: Server initialization ===")
print(f"=== RENDER STARTUP: PORT={os.environ.get('PORT')} ===")
print(f"=== RENDER STARTUP: Python version: {sys.version} ===")

# Import the FastAPI app from main.py
try:
    from main import app
    
    # This only runs when executed directly
    if __name__ == "__main__":
        import uvicorn
        port = int(os.environ.get("PORT", 10000))
        print(f"⚡ Starting server on port {port} ⚡")
        print(f"⚡ Using Python {sys.version} ⚡")
        print(f"⚡ Memory available: {os.environ.get('RENDER_MEMORY_TOTAL', 'unknown')} ⚡")
        
        # Add simple route handlers to overwrite any problematic ones
        @app.get("/api/minimal-chat")
        async def minimal_chat():
            return {"response": "This is a minimal chat response that doesn't use ML components"}
        
        uvicorn.run(app, host="0.0.0.0", port=port)
except Exception as e:
    print(f"CRITICAL ERROR DURING IMPORT: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)