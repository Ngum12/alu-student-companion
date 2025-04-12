import os
import sys

print("Pre-downloading models for Hugging Face deployment...")

# Set cache directory
os.environ["TRANSFORMERS_CACHE"] = "/tmp/model_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/model_cache"
os.environ["HF_HOME"] = "/tmp/model_cache"

# Create cache directory
os.makedirs("/tmp/model_cache", exist_ok=True)

# Configure extended timeouts
import urllib.request
import socket
socket.setdefaulttimeout(300)  # 5-minute timeout

try:
    # Pre-download the model
    from sentence_transformers import SentenceTransformer
    print("Downloading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/tmp/model_cache')
    print("✅ Model downloaded successfully!")
except Exception as e:
    print(f"⚠️ Error downloading model: {e}")
    sys.exit(1)