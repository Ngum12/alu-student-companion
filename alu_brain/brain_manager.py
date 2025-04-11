
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .search_engine import BrainSearchEngine
from .formatters import BrainResponseFormatter

class ALUBrainManager:
    """
    Manages the ALU Brain JSON knowledge base:
    - Loads and indexes JSON files
    - Provides search and retrieval functions
    - Formats responses for the prompt engine
    """
    
    def __init__(self, brain_dir: str = "alu_brain"):
        self.brain_dir = Path(brain_dir)
        self.knowledge_base = {}
        self.search_engine = BrainSearchEngine()
        self.formatter = BrainResponseFormatter()
        self.load_brain()
    
    def load_brain(self):
        """Load all JSON files from the alu_brain directory"""
        if not self.brain_dir.exists():
            print(f"Warning: ALU Brain directory not found at {self.brain_dir}")
            return
            
        json_files = list(self.brain_dir.glob("*.json"))
        if not json_files:
            print(f"Warning: No JSON files found in {self.brain_dir}")
            return
            
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    category = data.get('category')
                    if category and 'entries' in data:
                        self.knowledge_base[category] = data
                        print(f"Loaded {len(data['entries'])} entries from {json_path.name}")
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search in the knowledge base using the search engine"""
        return self.search_engine.search(query, self.knowledge_base, top_k)
    
    def get_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific entry by its ID"""
        return self.search_engine.get_entry_by_id(entry_id, self.knowledge_base)
    
    def get_entries_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get entries from a specific category"""
        return self.search_engine.get_entries_by_category(category, self.knowledge_base, limit)
    
    def format_for_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as context for the prompt engine"""
        return self.formatter.format_for_context(results)
