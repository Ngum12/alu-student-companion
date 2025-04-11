import os
from sentence_transformers import SentenceTransformer

# Set environment variable to only download CPU models
os.environ["FORCE_CPU"] = "1"

# Force minimal downloads
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/sentence_transformers"

# Only download what you need (no extras)
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_CACHE_MAX_SIZE"] = "2G"

print("Pre-caching sentence transformer model...")
# This will download the model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model cached successfully!")