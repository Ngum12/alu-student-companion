import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# For vector storage and retrieval
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Check for lightweight mode
LIGHTWEIGHT_MODE = os.environ.get("USE_LIGHTWEIGHT_MODE", "False").lower() == "true"

# Get data directory path from environment or use default
base_dir = os.environ.get("DATA_DIR", "./data")
alu_brain_dir = os.environ.get("ALU_BRAIN_DIR", "./alu_brain")

# Create proper paths based on environment
if os.environ.get("DEPLOYMENT_ENV") == "huggingface":
    # When deployed to Hugging Face
    DATA_DIR = Path(base_dir)
    ALU_BRAIN_DIR = Path(alu_brain_dir)
else:
    # Local development
    DATA_DIR = Path("./data") if not Path("./backend/data").exists() else Path("./backend/data")
    ALU_BRAIN_DIR = Path("./alu_brain") if not Path("./backend/alu_brain").exists() else Path("./backend/alu_brain")

# Set all derived paths
DOCUMENTS_DIR = DATA_DIR / "documents"
METADATA_FILE = DATA_DIR / "document_metadata.json"
VECTOR_DB_DIR = DATA_DIR / "vectordb"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Maximum chunk size for document splitting
MAX_CHUNK_SIZE = 1000  # characters
MAX_CHUNK_OVERLAP = 200  # characters

# Create directories if they don't exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

class Document:
    """Simple document class to store text and metadata"""
    def __init__(self, text: str, metadata: Dict[str, Any], score: Optional[float] = None):
        self.text = text
        self.metadata = metadata
        self.score = score

class RetrievalEngine:
    """
    Handles document retrieval using vector embeddings:
    - Document chunking and embedding
    - Semantic search
    - Context retrieval for the prompt engine
    """

    def __init__(self):
        # Initialize the embedding model
        self.embedding_model = self.load_embedding_model()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        
        # Create or get the collection
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        
        try:
            self.collection = self.client.get_collection(
                name="alu_documents",
                embedding_function=self.embedding_function
            )
            print("Connected to existing vector collection")
        except ValueError:
            self.collection = self.client.create_collection(
                name="alu_documents",
                embedding_function=self.embedding_function
            )
            print("Created new vector collection")
        
        # Initialize document processor reference
        from document_processor import DocumentProcessor
        self.document_processor = DocumentProcessor()

    def load_embedding_model(self):
        if LIGHTWEIGHT_MODE:
            # Use a smaller model for HF deployment
            return SentenceTransformer('paraphrase-MiniLM-L3-v2')
        else:
            # Your regular model
            return SentenceTransformer('all-MiniLM-L6-v2')

    def _chunk_text(self, text: str, chunk_size: int = MAX_CHUNK_SIZE, chunk_overlap: int = MAX_CHUNK_OVERLAP) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # If we're not at the end, try to find a good breakpoint
            if end < len(text):
                # Look for a period, newline, or space to break at
                breakpoints = [text.rfind('.', start, end), 
                              text.rfind('\n', start, end),
                              text.rfind(' ', start, end)]
                
                # Filter valid breakpoints and find the latest one
                valid_breakpoints = [bp for bp in breakpoints if bp != -1]
                if valid_breakpoints:
                    end = max(valid_breakpoints) + 1
            
            # Add this chunk to the list
            chunks.append(text[start:end])
            
            # Move the start point, accounting for overlap
            start = end - chunk_overlap if end - chunk_overlap > start else end
        
        return chunks

    def update_vector_store(self, doc_id: str):
        """Process and add a document to the vector store"""
        try:
            # Get document text
            doc_text = self.document_processor.get_document_text(doc_id)
            if not doc_text:
                print(f"Document text not found for ID: {doc_id}")
                return False
            
            # Get document metadata
            with open(METADATA_FILE, "r") as f:
                all_metadata = json.load(f)
            
            if doc_id not in all_metadata:
                print(f"Document metadata not found for ID: {doc_id}")
                return False
                
            metadata = all_metadata[doc_id]
            
            # Chunk the document
            chunks = self._chunk_text(doc_text)
            
            # Prepare for batch add
            doc_ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                doc_ids.append(chunk_id)
                
                # Create metadata for the chunk
                chunk_metadata = {
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "title": metadata.get("title", "Untitled"),
                    "source": metadata.get("source", "Unknown"),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                metadatas.append(chunk_metadata)
            
            # Add to the collection
            if chunks:
                self.collection.add(
                    documents=chunks,
                    ids=doc_ids,
                    metadatas=metadatas
                )
                print(f"Added {len(chunks)} chunks from document {doc_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error updating vector store: {e}")
            return False

    def remove_document(self, doc_id: str):
        """Remove a document's chunks from the vector store"""
        try:
            # Query to find all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            # Delete the chunks
            if results and results.get("ids"):
                for chunk_id in results["ids"]:
                    self.collection.delete(chunk_id)
                
                print(f"Removed {len(results['ids'])} chunks for document {doc_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error removing document from vector store: {e}")
            return False

    def rebuild_index(self):
        """Rebuild the entire vector index from scratch"""
        try:
            # Clear the collection
            self.collection.delete(where={})
            print("Cleared vector collection")
            
            # Get all document IDs
            with open(METADATA_FILE, "r") as f:
                all_metadata = json.load(f)
            
            # Add each document
            for doc_id in all_metadata.keys():
                self.update_vector_store(doc_id)
            
            print(f"Rebuilt vector index with {len(all_metadata)} documents")
            return True
            
        except Exception as e:
            print(f"Error rebuilding index: {e}")
            return False

    def retrieve_context(self, query: str, role: str = "student", top_k: int = 5) -> List[Document]:
        """
        Retrieve relevant context for a query:
        1. Perform semantic search against the vector store
        2. Return top matches as Document objects
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Create Document objects
            documents = []
            if results and results.get("documents") and results.get("documents")[0]:
                for i, doc_text in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else {}
                    
                    # Add distance/score if available
                    score = None
                    if results.get("distances") and results["distances"][0]:
                        score = results["distances"][0][i]
                    
                    documents.append(Document(
                        text=doc_text,
                        metadata=metadata,
                        score=score
                    ))
            
            # Role-based filtering (could be expanded)
            if role != "admin" and role != "faculty":
                # For regular students, filter out admin-only content if needed
                # (This is just a placeholder for role-based access logic)
                pass
                
            return documents
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
