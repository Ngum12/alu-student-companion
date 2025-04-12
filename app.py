import os

# Set environment variables for Hugging Face
os.environ["TRANSFORMERS_CACHE"] = "/tmp/model_cache"
os.environ["HF_HOME"] = "/tmp/model_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/model_cache"
os.environ["PYTHONUNBUFFERED"] = "1"

# Import the FastAPI app from main.py
from main import app

# This is needed for Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)  # Hugging Face uses port 7860