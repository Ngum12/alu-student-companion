
from retrieval_engine import RetrievalEngine, Document
from alu_brain import ALUBrainManager
from typing import List, Dict, Any, Optional
import time

class ExtendedRetrievalEngine(RetrievalEngine):
    """
    Extends the base RetrievalEngine to utilize the ALU Brain JSON knowledge base
    with enhanced integration between vector store and structured knowledge
    """
    
    def __init__(self):
        super().__init__()
        self.alu_brain = ALUBrainManager()
        self._cache = {}  # Simple in-memory cache
        self._cache_ttl = 300  # Cache TTL in seconds (5 minutes)
        print("Extended Retrieval Engine initialized with ALU Brain integration and performance optimizations")
    
    def retrieve_context(self, query: str, role: str = "student", **kwargs):
        """
        Enhanced retrieve_context that combines vector store results with ALU Brain results
        using intelligent merging based on relevance scores
        """
        # Check cache first for improved performance
        cache_key = f"{query}:{role}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
            
        # Get results from the original vector store (parent class)
        vector_results = super().retrieve_context(query, role, **kwargs)
        
        # Get results from ALU Brain
        brain_results = self.alu_brain.search(query, top_k=5)
        
        # Process ALU Brain results if available
        brain_documents = []
        if brain_results:
            brain_context = self.alu_brain.format_for_context(brain_results)
            
            for i, result in enumerate(brain_results):
                entry = result['entry']
                category = result['category']
                
                # Extract core information
                question = entry.get('question', '')
                answer = entry.get('answer', '')
                entry_type = entry.get('type', 'text')
                
                # Create text content based on entry type
                text = self._format_entry_content(entry, question, answer, entry_type)
                
                # Create Document object
                doc = Document(
                    text=text,
                    metadata={
                        'title': question or f"ALU {category.replace('_', ' ').title()} Knowledge",
                        'source': f"ALU Brain: {category.replace('_', ' ').title()}",
                        'type': entry_type,
                        'score': result.get('score', 0)
                    }
                )
                brain_documents.append(doc)
        
        # Intelligently merge vector and brain results
        merged_results = self._merge_results(vector_results, brain_documents)
        
        # Store in cache
        self._store_in_cache(cache_key, merged_results)
        
        return merged_results
    
    def _format_entry_content(self, entry, question, answer, entry_type):
        """Helper method to format entry content based on type"""
        if entry_type == 'link_response' and 'links' in entry:
            links_text = "\n".join([f"- {link.get('text', '')}: {link.get('url', '')}" 
                                  for link in entry.get('links', [])])
            return f"{question}\n\n{answer}\n\n{links_text}"
        
        elif entry_type == 'table_response' and 'table' in entry:
            table = entry.get('table', {})
            if 'headers' in table and 'rows' in table:
                # Simple text representation of the table
                table_text = "Table data:\n"
                for row in table.get('rows', []):
                    table_text += f"  {', '.join(row)}\n"
                return f"{question}\n\n{answer}\n\n{table_text}"
            else:
                return f"{question}\n\n{answer}"
        
        elif entry_type in ['statistical_response', 'date_response', 'procedural_response']:
            # For these types, include the specialized content
            special_content = ""
            if 'statistics' in entry:
                special_content = "\n".join([f"- {stat.get('metric', '')}: {stat.get('value', '')}" 
                                           for stat in entry.get('statistics', [])])
            elif 'dates' in entry:
                special_content = "\n".join([f"- {date.get('round', '')}: {date.get('deadline', '')}" 
                                           for date in entry.get('dates', [])])
            elif 'steps' in entry:
                steps = entry.get('steps', [])
                special_content = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
            
            return f"{question}\n\n{answer}\n\n{special_content}"
        
        else:
            # Default format for text responses
            return f"{question}\n\n{answer}"
    
    def _merge_results(self, vector_results: List[Document], brain_results: List[Document]) -> List[Document]:
        """
        Intelligently merge vector and brain results based on relevance and diversity
        """
        # Start with all vector results
        all_results = list(vector_results)
        
        # If no brain results, just return vector results
        if not brain_results:
            return all_results
            
        # If no vector results, just return brain results
        if not vector_results:
            return brain_results
        
        # Sort both lists by score
        vector_results.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        brain_results.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
        
        # Take the top result from brain results and insert at position 2
        if brain_results:
            top_brain = brain_results.pop(0)
            if len(all_results) >= 1:
                all_results.insert(1, top_brain)
            else:
                all_results.append(top_brain)
        
        # Interleave remaining results (optimized algorithm)
        remaining = []
        while brain_results and len(remaining) < 8:  # Limit to prevent too many results
            # Take one from brain
            if brain_results:
                remaining.append(brain_results.pop(0))
            
        # Add remaining interleaved results
        i = 2  # Start after the first vector result and the top brain result
        for doc in remaining:
            if i < len(all_results):
                all_results.insert(i, doc)
                i += 2  # Skip every other position
            else:
                all_results.append(doc)
        
        # Ensure we don't return too many results (limit to 10)
        return all_results[:10]
        
    def _get_from_cache(self, key):
        """Get result from cache if it exists and hasn't expired"""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self._cache_ttl:
                return entry['data']
        return None
        
    def _store_in_cache(self, key, data):
        """Store result in cache with current timestamp"""
        self._cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        # Clean up old cache entries
        self._clean_cache()
        
    def _clean_cache(self):
        """Remove expired cache entries"""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now - v['timestamp'] > self._cache_ttl]
        for key in expired_keys:
            del self._cache[key]
