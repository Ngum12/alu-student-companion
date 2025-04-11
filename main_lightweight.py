"""Lightweight wrapper for main.py that disables memory-intensive features"""
import os
import gc
import sys

# Set environment variables to reduce memory usage
os.environ["USE_LIGHTWEIGHT_MODE"] = "True" 
os.environ["DISABLE_VECTOR_SEARCH"] = "True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Prevent model downloads

# Run garbage collection
gc.collect()

# Import the actual app
from main import app

# Modify any app settings here if needed
# For example, you could modify the CORS settings or disable certain routes