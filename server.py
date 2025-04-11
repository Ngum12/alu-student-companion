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

# First import the minimal app (this will start fast)
try:
    from minimal_app import app
    print("✅ Minimal app loaded successfully, starting server...")
    
    # Then try to import the full app in a background thread
    import threading
    
    def load_full_app():
        try:
            print("🔄 Starting to load full app components...")
            import time
            start_time = time.time()
            
            # Import the full app
            from main import app as full_app
            
            # More aggressive - replace the entire app
            global app
            app = full_app
            
            elapsed = time.time() - start_time
            print(f"✅ Full app loaded successfully in {elapsed:.2f} seconds")
        except Exception as e:
            print(f"⚠️ Error loading full app: {e}")
    
    # Start loading full app in background
    bg_thread = threading.Thread(target=load_full_app)
    bg_thread.daemon = True
    bg_thread.start()
    
except Exception as e:
    print(f"⚠️ CRITICAL ERROR: {e}")
    sys.exit(1)

# This only runs when executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"⚡ Starting server on port {port} ⚡")
    
    uvicorn.run(app, host="0.0.0.0", port=port)