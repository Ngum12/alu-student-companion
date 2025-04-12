import os
print("Pre-downloading models for offline use...")

# Set cache directory
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"

# Create cache directory
os.makedirs("./model_cache", exist_ok=True)

# Pre-download models with extended timeout
from sentence_transformers import SentenceTransformer
import requests.adapters
import urllib3

# Extend timeout for downloads
urllib3.util.timeout.Timeout._DEFAULT_TIMEOUT = 120
adapter = requests.adapters.HTTPAdapter(max_retries=5)
session = requests.Session()
session.mount('https://', adapter)

# Download the model
print("Downloading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model_cache')
print("✅ Model download complete!")