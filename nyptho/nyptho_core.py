
import json
import time
import random
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

class NypthoCore:
    """
    Nyptho: A meta-learning system designed to learn from other AI models
    - Aggregates and processes responses from other AI systems
    - Builds a response model based on observed patterns
    - Can generate responses emulating learned patterns
    """
    
    def __init__(self, 
                 model_dir: str = "./nyptho_models",
                 learning_rate: float = 0.01,
                 memory_size: int = 1000):
        """Initialize the Nyptho core system"""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.interaction_memory = []
        self.pattern_database = {}
        self.response_templates = {}
        self.personality_traits = {
            "helpfulness": 0.8,
            "creativity": 0.7, 
            "precision": 0.9,
            "adaptability": 0.8,
            "friendliness": 0.75
        }
        
        # Load any existing models
        self._load_models()
        print("Nyptho Core System initialized")
    
    def observe_interaction(self, 
                           query: str, 
                           response: str, 
                           source_model: str,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record and learn from an AI interaction
        
        Args:
            query: User query that generated the response
            response: The AI's response to learn from
            source_model: Identifier of the source AI model
            metadata: Additional information about the interaction
        """
        # Create interaction record
        interaction = {
            "query": query,
            "response": response,
            "source_model": source_model,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "features": self._extract_features(query, response)
        }
        
        # Add to memory (with size limit)
        self.interaction_memory.append(interaction)
        if len(self.interaction_memory) > self.memory_size:
            self.interaction_memory.pop(0)
        
        # Update pattern database
        self._update_patterns(interaction)
        
        # Periodic saving of learned models
        if len(self.interaction_memory) % 50 == 0:
            self._save_models()
    
    def generate_response(self, 
                         query: str, 
                         context: Optional[List[Dict[str, Any]]] = None,
                         persona: Optional[Dict[str, float]] = None) -> str:
        """
        Generate a response based on learned patterns
        
        Args:
            query: User query to respond to
            context: Additional context for the response
            persona: Personality traits to use (overrides default)
        
        Returns:
            Generated response based on learned patterns
        """
        if not self.pattern_database:
            return "I'm still learning and don't have enough information to provide a useful response yet."
        
        # Use provided persona or default
        active_persona = persona or self.personality_traits
        
        # Find matching patterns
        matching_patterns = self._find_matching_patterns(query)
        
        if not matching_patterns:
            # Fallback to generic response
            return self._generate_generic_response(query, active_persona)
        
        # Generate response based on patterns and persona
        responses = []
        weights = []
        
        for pattern in matching_patterns[:5]:  # Use top 5 matches
            template = pattern["template"]
            confidence = pattern["confidence"]
            
            # Generate response from template
            response = self._apply_template(template, query, context, active_persona)
            responses.append(response)
            weights.append(confidence)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
            
            # Weighted selection of response
            selected_idx = np.random.choice(len(responses), p=weights)
            return responses[selected_idx]
        
        return self._generate_generic_response(query, active_persona)
    
    def set_personality(self, traits: Dict[str, float]) -> None:
        """Update personality traits"""
        for trait, value in traits.items():
            if trait in self.personality_traits:
                self.personality_traits[trait] = max(0.0, min(1.0, value))
    
    def _extract_features(self, query: str, response: str) -> Dict[str, Any]:
        """Extract features from query and response for pattern learning"""
        features = {
            "query_length": len(query),
            "response_length": len(response),
            "query_keywords": self._extract_keywords(query),
            "response_structure": self._analyze_structure(response),
            "sentiment": self._analyze_sentiment(query)
        }
        return features
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple keyword extraction
        words = text.lower().split()
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'with', 'by', 'at', 'on'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of a response"""
        structure = {
            "has_greeting": any(greeting in text.lower() for greeting in ["hello", "hi", "greetings", "hey"]),
            "has_question": "?" in text,
            "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
            "sentences": text.count(".") + text.count("!") + text.count("?"),
        }
        return structure
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        positive_words = {"good", "great", "awesome", "excellent", "fantastic", "wonderful", "amazing", "happy", "love", "like"}
        negative_words = {"bad", "terrible", "awful", "horrible", "poor", "sad", "hate", "dislike", "worst", "failure"}
        
        words = set(text.lower().split())
        pos_count = len(positive_words.intersection(words))
        neg_count = len(negative_words.intersection(words))
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"
    
    def _update_patterns(self, interaction: Dict[str, Any]) -> None:
        """Update pattern database with new interaction"""
        query = interaction["query"]
        response = interaction["response"]
        features = interaction["features"]
        
        # Extract keywords for pattern matching
        keywords = features["query_keywords"]
        
        # Create pattern key from keywords
        for keyword in keywords:
            if keyword not in self.pattern_database:
                self.pattern_database[keyword] = []
            
            # Add response template
            template = self._create_template(query, response)
            
            # Check if similar template exists
            existing = False
            for pattern in self.pattern_database[keyword]:
                if self._template_similarity(pattern["template"], template) > 0.7:
                    # Update existing pattern
                    pattern["count"] += 1
                    pattern["confidence"] = min(0.9, pattern["confidence"] + 0.05)
                    existing = True
                    break
            
            if not existing:
                # Add new pattern
                self.pattern_database[keyword].append({
                    "template": template,
                    "source": interaction["source_model"],
                    "count": 1,
                    "confidence": 0.5,  # Initial confidence
                    "features": features
                })
    
    def _create_template(self, query: str, response: str) -> Dict[str, Any]:
        """Create a response template from query-response pair"""
        template = {
            "base_text": response,
            "query_pattern": self._generalize_query(query),
            "placeholders": self._identify_placeholders(query, response),
            "structure": self._analyze_structure(response)
        }
        return template
    
    def _generalize_query(self, query: str) -> str:
        """Create a generalized pattern from a query"""
        # Very simple generalization for now
        # In a real system, this would use NLP to identify entities and concepts
        words = query.lower().split()
        generalized = []
        
        for word in words:
            if word in ["who", "what", "when", "where", "why", "how"]:
                generalized.append(f"<{word}>")
            elif word.isdigit():
                generalized.append("<number>")
            else:
                generalized.append(word)
        
        return " ".join(generalized)
    
    def _identify_placeholders(self, query: str, response: str) -> Dict[str, str]:
        """Identify potential placeholders in the response"""
        # This is a placeholder for more advanced NLP-based extraction
        # In a real system, this would identify query terms that appear in the response
        placeholders = {}
        query_terms = set(query.lower().split())
        
        for term in query_terms:
            if term in response.lower() and len(term) > 3:
                placeholders[term] = f"<{term}>"
        
        return placeholders
    
    def _template_similarity(self, template1: Dict[str, Any], template2: Dict[str, Any]) -> float:
        """Calculate similarity between two templates"""
        # Simple similarity based on structure
        structure1 = template1["structure"]
        structure2 = template2["structure"]
        
        # Count matching attributes
        matches = 0
        total = 0
        
        for key in structure1:
            if key in structure2:
                total += 1
                if structure1[key] == structure2[key]:
                    matches += 1
        
        return matches / max(1, total)
    
    def _find_matching_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Find patterns that match a query"""
        keywords = self._extract_keywords(query)
        matches = []
        
        # Gather all potential matches from keywords
        for keyword in keywords:
            if keyword in self.pattern_database:
                matches.extend(self.pattern_database[keyword])
        
        # Sort by confidence and count
        matches.sort(key=lambda x: (x["confidence"], x["count"]), reverse=True)
        return matches
    
    def _apply_template(self, template: Dict[str, Any], query: str, context: Optional[List], persona: Dict[str, float]) -> str:
        """Apply a template to generate a response"""
        base_text = template["base_text"]
        
        # Apply personality traits
        if persona["creativity"] > 0.7:
            # Add some variation
            base_text = self._add_variation(base_text)
        
        # Replace placeholders
        for original, placeholder in template["placeholders"].items():
            if original in query.lower():
                base_text = base_text.replace(placeholder, original)
        
        # Add precision details if needed
        if persona["precision"] > 0.8 and context:
            base_text = self._add_details(base_text, context)
        
        # Add friendliness markers if needed
        if persona["friendliness"] > 0.7:
            base_text = self._add_friendliness(base_text)
        
        return base_text
    
    def _add_variation(self, text: str) -> str:
        """Add variation to a response for creativity"""
        variations = [
            "I think ", 
            "In my view, ", 
            "Based on my analysis, ",
            "From what I understand, ",
            "It appears that "
        ]
        
        sentences = text.split(". ")
        if len(sentences) > 1:
            idx = random.randint(0, len(sentences)-1)
            variation = random.choice(variations)
            sentences[idx] = variation + sentences[idx][0].lower() + sentences[idx][1:]
        
        return ". ".join(sentences)
    
    def _add_details(self, text: str, context: List) -> str:
        """Add details from context for precision"""
        if not context:
            return text
            
        # Get a relevant detail from context
        detail = context[0].get("text", "") if context and len(context) > 0 else ""
        
        if detail:
            return f"{text}\n\nAdditionally: {detail}"
        return text
    
    def _add_friendliness(self, text: str) -> str:
        """Add friendliness markers"""
        friendly_openings = [
            "Happy to help! ",
            "Great question! ",
            "I'd be delighted to assist. ",
            "Thanks for asking. "
        ]
        
        friendly_closings = [
            "\n\nHope that helps!",
            "\n\nLet me know if you need anything else.",
            "\n\nIs there anything else you'd like to know?"
        ]
        
        # Only add opening if there isn't one already
        if not any(text.startswith(opening) for opening in ["Hi", "Hello", "Hey", "Thanks", "Thank", "Great", "Happy"]):
            text = random.choice(friendly_openings) + text
        
        # Only add closing if there isn't one already
        if not any(text.endswith(closing.strip()) for closing in friendly_closings):
            text = text + random.choice(friendly_closings)
        
        return text
    
    def _generate_generic_response(self, query: str, persona: Dict[str, float]) -> str:
        """Generate a generic response when no pattern matches"""
        generic_templates = [
            "I'm still learning about topics like this. Could you tell me more about what you're looking for regarding {topic}?",
            "That's an interesting question about {topic}. I'm gathering more information on this subject.",
            "I don't have enough information about {topic} yet, but I'm continuously learning.",
            "I'm not fully trained on {topic} yet, but I'm interested in learning more about your question."
        ]
        
        # Extract a topic from the query
        keywords = self._extract_keywords(query)
        topic = keywords[0] if keywords else "this topic"
        
        template = random.choice(generic_templates)
        response = template.format(topic=topic)
        
        # Apply personality
        if persona["friendliness"] > 0.7:
            response = self._add_friendliness(response)
        
        return response
    
    def _save_models(self) -> None:
        """Save learned models to disk"""
        try:
            model_data = {
                "pattern_database": self.pattern_database,
                "personality_traits": self.personality_traits,
                "version": "0.1",
                "timestamp": time.time()
            }
            
            model_path = self.model_dir / "nyptho_model.json"
            with open(model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
                
            print(f"Model saved to {model_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _load_models(self) -> None:
        """Load learned models from disk"""
        try:
            model_path = self.model_dir / "nyptho_model.json"
            
            if model_path.exists():
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                    
                if "pattern_database" in model_data:
                    self.pattern_database = model_data["pattern_database"]
                    
                if "personality_traits" in model_data:
                    self.personality_traits = model_data["personality_traits"]
                    
                print(f"Loaded model from {model_path}")
                print(f"Loaded {len(self.pattern_database)} pattern categories")
            
        except Exception as e:
            print(f"Error loading model: {e}")
