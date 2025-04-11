
import os
import uuid
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import UploadFile, HTTPException
import time

# Document handling libraries
import pypdf
import docx2txt

# Create data directories if they don't exist
DATA_DIR = Path("./data")
DOCUMENTS_DIR = DATA_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_FILE = DATA_DIR / "document_metadata.json"

class DocumentProcessor:
    """
    Handles document processing including:
    - Uploading and storing documents
    - Extracting text from different document formats
    - Maintaining document metadata
    """

    def __init__(self):
        self.supported_formats = {
            "application/pdf": self._extract_pdf_text,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": self._extract_docx_text,
            "text/plain": self._extract_text_file,
            "text/markdown": self._extract_text_file,
        }
        self._initialize_metadata()

    def _initialize_metadata(self):
        """Initialize metadata storage"""
        if not METADATA_FILE.exists():
            with open(METADATA_FILE, "w") as f:
                json.dump({}, f)

    async def process_document(self, file: UploadFile, title: Optional[str] = None, source: str = "user-upload") -> str:
        """
        Process an uploaded document:
        1. Extract text based on file type
        2. Save document metadata
        3. Return document ID for further processing
        """
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Determine document format and extract text
        content_type = file.content_type
        if content_type not in self.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {content_type}. Supported formats: {', '.join(self.supported_formats.keys())}"
            )

        # Save the original file
        file_path = DOCUMENTS_DIR / f"{doc_id}{Path(file.filename).suffix}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Reset file position
        await file.seek(0)
        
        # Extract text
        text = await self.supported_formats[content_type](file)
        
        # If title is not provided, use the filename without extension
        if not title:
            title = Path(file.filename).stem
        
        # Save text to a file
        text_file_path = DOCUMENTS_DIR / f"{doc_id}.txt"
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Update metadata
        metadata = {
            "id": doc_id,
            "title": title,
            "filename": file.filename,
            "content_type": content_type,
            "source": source,
            "length": len(text),
            "upload_time": time.time(),
            "original_file": str(file_path),
            "text_file": str(text_file_path)
        }
        
        self._save_metadata(doc_id, metadata)
        
        return doc_id

    async def _extract_pdf_text(self, file: UploadFile) -> str:
        """Extract text from PDF files"""
        temp_path = DOCUMENTS_DIR / f"temp_{uuid.uuid4()}.pdf"
        try:
            # Save to temporary file
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Process PDF using pypdf
            text = ""
            with open(temp_path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
            
            return text
        finally:
            # Clean up temp file
            if temp_path.exists():
                os.remove(temp_path)

    async def _extract_docx_text(self, file: UploadFile) -> str:
        """Extract text from DOCX files"""
        temp_path = DOCUMENTS_DIR / f"temp_{uuid.uuid4()}.docx"
        try:
            # Save to temporary file
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Process DOCX using docx2txt
            text = docx2txt.process(temp_path)
            return text
        finally:
            # Clean up temp file
            if temp_path.exists():
                os.remove(temp_path)

    async def _extract_text_file(self, file: UploadFile) -> str:
        """Extract text from plain text files"""
        content = await file.read()
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            # Try different encoding if UTF-8 fails
            return content.decode("latin-1")

    def _save_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """Save document metadata to the metadata file"""
        try:
            # Load existing metadata
            with open(METADATA_FILE, "r") as f:
                all_metadata = json.load(f)
            
            # Add new metadata
            all_metadata[doc_id] = metadata
            
            # Save updated metadata
            with open(METADATA_FILE, "w") as f:
                json.dump(all_metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error saving metadata: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving document metadata: {str(e)}")

    def list_documents(self) -> List[Dict[str, Any]]:
        """Get a list of all document metadata"""
        try:
            with open(METADATA_FILE, "r") as f:
                all_metadata = json.load(f)
            
            # Return a list of metadata objects
            return list(all_metadata.values())
            
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its metadata"""
        try:
            # Load existing metadata
            with open(METADATA_FILE, "r") as f:
                all_metadata = json.load(f)
            
            # Check if document exists
            if doc_id not in all_metadata:
                return False
            
            # Get metadata for the document
            metadata = all_metadata[doc_id]
            
            # Delete the original file if it exists
            original_file = Path(metadata.get("original_file", ""))
            if original_file.exists():
                os.remove(original_file)
            
            # Delete the text file if it exists
            text_file = Path(metadata.get("text_file", ""))
            if text_file.exists():
                os.remove(text_file)
            
            # Remove from metadata
            del all_metadata[doc_id]
            
            # Save updated metadata
            with open(METADATA_FILE, "w") as f:
                json.dump(all_metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get the extracted text for a document"""
        try:
            # Load metadata
            with open(METADATA_FILE, "r") as f:
                all_metadata = json.load(f)
            
            # Check if document exists
            if doc_id not in all_metadata:
                return None
            
            # Get text file path
            text_file = Path(all_metadata[doc_id].get("text_file", ""))
            if not text_file.exists():
                return None
            
            # Read text file
            with open(text_file, "r", encoding="utf-8") as f:
                return f.read()
                
        except Exception as e:
            print(f"Error getting document text: {e}")
            return None
