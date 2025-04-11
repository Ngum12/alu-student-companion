
import random
from typing import List, Dict, Any, Optional, Tuple
import time
import json
from pathlib import Path

class KnowledgeDistiller:
    """
    Knowledge Distillation module for Nyptho:
    - Extracts and consolidates knowledge from model responses
    - Identifies knowledge gaps
    - Creates structured knowledge representations
    """
    
    def __init__(self, 
                knowledge_dir: str = "./nyptho_knowledge",
                confidence_threshold: float = 0.7):
        """Initialize the knowledge distiller"""
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        self.confidence_threshold = confidence_threshold
        self.knowledge_base = {}
        self.knowledge_gaps = set()
        self.topic_map = {}
        self.consistency_metrics = {}
        
        self._load_knowledge_base()
    
    def process_interaction(self, 
                           query: str, 
                           response: str,
                           source_model: str,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Process an interaction to extract knowledge
        
        Args:
            query: User query
            response: AI response
            source_model: The AI model that generated the response
            metadata: Additional interaction metadata
        """
        # Extract knowledge claims from the response
        knowledge_claims = self._extract_knowledge_claims(response)
        
        # Update knowledge base with extracted claims
        for claim in knowledge_claims:
            self._add_knowledge_claim(claim, query, source_model)
        
        # Identify potential knowledge gaps from the query
        gaps = self._identify_knowledge_gaps(query, response)
        for gap in gaps:
            self.knowledge_gaps.add(gap)
        
        # Update topic mapping
        topics = self._extract_topics(query, response)
        for topic in topics:
            if topic not in self.topic_map:
                self.topic_map[topic] = set()
            
            # Link related claims to this topic
            for claim in knowledge_claims:
                if any(term in claim.lower() for term in topic.lower().split()):
                    self.topic_map[topic].add(claim[:50])  # Store first 50 chars as identifier
        
        # Periodically save knowledge base
        if random.random() < 0.1:  # 10% chance on each interaction
            self._save_knowledge_base()
    
    def query_knowledge(self, 
                       query: str, 
                       confidence_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Query the knowledge base
        
        Args:
            query: Knowledge query
            confidence_threshold: Minimum confidence level (overrides default)
            
        Returns:
            List of relevant knowledge items
        """
        threshold = confidence_threshold or self.confidence_threshold
        
        # Extract query terms
        terms = set(query.lower().split())
        results = []
        
        # First look for direct matches in topic map
        direct_matches = []
        for topic in self.topic_map:
            topic_terms = set(topic.lower().split())
            if terms.intersection(topic_terms):
                direct_matches.extend(list(self.topic_map[topic]))
        
        # Then search through knowledge base
        for claim_key, claim_data in self.knowledge_base.items():
            # Skip if confidence too low
            if claim_data["confidence"] < threshold:
                continue
                
            # Check if this is a direct match from topic map
            if claim_key[:50] in direct_matches:
                results.append({
                    "claim": claim_key,
                    "sources": claim_data["sources"],
                    "confidence": claim_data["confidence"],
                    "last_updated": claim_data["last_updated"],
                    "relevance": 1.0  # Direct topic match
                })
                continue
            
            # Otherwise check term overlap
            claim_terms = set(claim_key.lower().split())
            overlap = terms.intersection(claim_terms)
            
            if overlap:
                relevance = len(overlap) / max(len(terms), len(claim_terms))
                
                if relevance > 0.2:  # Minimum relevance threshold
                    results.append({
                        "claim": claim_key,
                        "sources": claim_data["sources"],
                        "confidence": claim_data["confidence"],
                        "last_updated": claim_data["last_updated"],
                        "relevance": relevance
                    })
        
        # Sort by combined relevance and confidence
        results.sort(key=lambda x: (x["relevance"] * 0.7 + x["confidence"] * 0.3), reverse=True)
        return results[:10]  # Return top 10
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            "total_claims": len(self.knowledge_base),
            "high_confidence_claims": sum(1 for c in self.knowledge_base.values() if c["confidence"] > 0.8),
            "knowledge_gaps": len(self.knowledge_gaps),
            "topics": len(self.topic_map),
            "source_distribution": self._get_source_distribution(),
            "top_topics": self._get_top_topics(10)
        }
    
    def _extract_knowledge_claims(self, response: str) -> List[str]:
        """Extract factual claims from response text"""
        # Split by sentence
        claims = []
        sentences = response.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Skip uncertain statements
            uncertainty_markers = ["might", "may", "could", "perhaps", "possibly", "I think", "I believe"]
            if any(marker in sentence.lower() for marker in uncertainty_markers):
                continue
                
            # Skip questions
            if sentence.endswith('?'):
                continue
                
            # Skip first-person statements
            if sentence.lower().startswith('i '):
                continue
                
            # Skip imperative sentences
            imperative_starters = ["let", "please", "consider", "note", "remember"]
            if any(sentence.lower().startswith(starter) for starter in imperative_starters):
                continue
            
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
                
            # Add period if needed
            if not sentence.endswith('.'):
                sentence += '.'
                
            claims.append(sentence)
        
        return claims
    
    def _add_knowledge_claim(self, claim: str, query: str, source_model: str) -> None:
        """Add or update a knowledge claim"""
        # Normalize claim for storage
        norm_claim = claim.strip()
        
        # Create or update
        if norm_claim in self.knowledge_base:
            kb_item = self.knowledge_base[norm_claim]
            
            # Update confidence based on repeated observations
            kb_item["confidence"] = min(0.95, kb_item["confidence"] + 0.05)
            
            # Add source if new
            source_key = f"{source_model}:{query[:20]}"
            if source_key not in kb_item["sources"]:
                kb_item["sources"].append(source_key)
                
            kb_item["occurrences"] += 1
            kb_item["last_updated"] = time.time()
            
        else:
            # Create new entry
            self.knowledge_base[norm_claim] = {
                "confidence": 0.6,  # Initial confidence
                "sources": [f"{source_model}:{query[:20]}"],
                "occurrences": 1,
                "last_updated": time.time(),
                "created": time.time()
            }
    
    def _identify_knowledge_gaps(self, query: str, response: str) -> List[str]:
        """Identify potential knowledge gaps from query and response"""
        gaps = []
        
        # Check for uncertainty markers in the response
        uncertainty_markers = [
            "I don't have information", 
            "I'm not sure",
            "I cannot provide",
            "I don't know", 
            "insufficient information",
            "no data on",
            "not enough context"
        ]
        
        if any(marker in response for marker in uncertainty_markers):
            # Extract query topics as potential gaps
            topics = self._extract_topics(query, "")
            gaps.extend(topics)
            
        return gaps
    
    def _extract_topics(self, query: str, response: str) -> List[str]:
        """Extract potential topics from query and response"""
        topics = []
        
        # Get potential topics from query
        query_words = query.lower().split()
        
        # Filter out common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 
                     'was', 'were', 'is', 'be', 'been', 'being', 'have', 
                     'has', 'had', 'to', 'for', 'of', 'by', 'with'}
        
        filtered_words = [word for word in query_words if word not in stop_words and len(word) > 3]
        
        # Look for pairs of meaningful words as topics
        for i in range(len(filtered_words) - 1):
            if len(filtered_words[i]) > 3 and len(filtered_words[i+1]) > 3:
                topic = f"{filtered_words[i]} {filtered_words[i+1]}"
                topics.append(topic)
        
        # Add individual words as topics
        for word in filtered_words:
            if word not in [t.split()[0] for t in topics]:  # Avoid duplicates
                topics.append(word)
        
        return topics
    
    def _get_source_distribution(self) -> Dict[str, int]:
        """Get distribution of knowledge by source model"""
        sources = {}
        
        for claim_data in self.knowledge_base.values():
            for source in claim_data["sources"]:
                model = source.split(':', 1)[0]
                if model not in sources:
                    sources[model] = 0
                sources[model] += 1
                
        return sources
    
    def _get_top_topics(self, n: int) -> List[Tuple[str, int]]:
        """Get top topics by number of claims"""
        topic_counts = []
        
        for topic, claims in self.topic_map.items():
            topic_counts.append((topic, len(claims)))
            
        topic_counts.sort(key=lambda x: x[1], reverse=True)
        return topic_counts[:n]
    
    def _save_knowledge_base(self) -> None:
        """Save knowledge base to disk"""
        try:
            # Save knowledge base
            kb_path = self.knowledge_dir / "knowledge_base.json"
            with open(kb_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            
            # Save topic map
            topic_path = self.knowledge_dir / "topic_map.json"
            topic_data = {topic: list(claims) for topic, claims in self.topic_map.items()}
            with open(topic_path, 'w') as f:
                json.dump(topic_data, f, indent=2)
            
            # Save knowledge gaps
            gaps_path = self.knowledge_dir / "knowledge_gaps.json"
            with open(gaps_path, 'w') as f:
                json.dump(list(self.knowledge_gaps), f, indent=2)
                
            print(f"Knowledge base saved to {self.knowledge_dir}")
            
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    def _load_knowledge_base(self) -> None:
        """Load knowledge base from disk"""
        try:
            # Load knowledge base
            kb_path = self.knowledge_dir / "knowledge_base.json"
            if kb_path.exists():
                with open(kb_path, 'r') as f:
                    self.knowledge_base = json.load(f)
            
            # Load topic map
            topic_path = self.knowledge_dir / "topic_map.json"
            if topic_path.exists():
                with open(topic_path, 'r') as f:
                    topic_data = json.load(f)
                    self.topic_map = {topic: set(claims) for topic, claims in topic_data.items()}
            
            # Load knowledge gaps
            gaps_path = self.knowledge_dir / "knowledge_gaps.json"
            if gaps_path.exists():
                with open(gaps_path, 'r') as f:
                    self.knowledge_gaps = set(json.load(f))
                    
            print(f"Loaded knowledge base with {len(self.knowledge_base)} claims")
            
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
