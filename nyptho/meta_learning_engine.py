
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import random
import time

class MetaLearningEngine:
    """
    Meta-Learning Engine for Nyptho:
    - Observes and analyzes responses from different AI models
    - Extracts patterns, techniques, and strategies
    - Builds a meta-model of effective response generation
    """
    
    def __init__(self, nyptho_core):
        self.nyptho_core = nyptho_core
        self.learning_factors = {
            "pattern_recognition": 0.7,
            "response_structure": 0.8,
            "style_adaptation": 0.6,
            "context_awareness": 0.75
        }
        self.observation_count = 0
        self.learning_thresholds = [50, 100, 250, 500, 1000]
        self.current_learning_level = 0
        
        # Track model performance metrics
        self.model_performance = {}
    
    def observe_model_interaction(self, 
                                 model_id: str,
                                 query: str,
                                 response: str,
                                 context: Optional[List[Dict[str, Any]]] = None,
                                 feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Process an observation of another AI model's response
        
        Args:
            model_id: Identifier of the observed AI model
            query: User query
            response: Model's response
            context: The context provided for the response
            feedback: Any feedback on the response quality
        """
        # Pass to core
        self.nyptho_core.observe_interaction(
            query=query,
            response=response,
            source_model=model_id,
            metadata={
                "context": context,
                "feedback": feedback
            }
        )
        
        # Extract learning features
        features = self._extract_learning_features(query, response, context)
        
        # Update model performance tracking
        if model_id not in self.model_performance:
            self.model_performance[model_id] = {
                "observations": 0,
                "avg_response_length": 0,
                "response_patterns": {},
                "context_usage": 0
            }
        
        perf = self.model_performance[model_id]
        perf["observations"] += 1
        perf["avg_response_length"] = ((perf["avg_response_length"] * (perf["observations"] - 1)) + 
                                     len(response)) / perf["observations"]
        
        # Track context usage
        if context and features["context_references"] > 0:
            perf["context_usage"] += 1
        
        # Track response patterns
        for pattern in features["identified_patterns"]:
            if pattern not in perf["response_patterns"]:
                perf["response_patterns"][pattern] = 0
            perf["response_patterns"][pattern] += 1
        
        # Update overall observation count
        self.observation_count += 1
        
        # Check if we've reached a learning threshold
        self._check_learning_progression()
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Compare the performance and patterns of different AI models
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            Comparative analysis of the models
        """
        results = {
            "model_comparisons": {},
            "overall_ranking": [],
            "unique_strengths": {}
        }
        
        valid_models = [m for m in model_ids if m in self.model_performance]
        
        if not valid_models:
            return {"error": "No valid models to compare"}
        
        # Create comparison metrics
        for model_id in valid_models:
            perf = self.model_performance[model_id]
            
            results["model_comparisons"][model_id] = {
                "observations": perf["observations"],
                "avg_response_length": perf["avg_response_length"],
                "context_usage_rate": perf["context_usage"] / max(1, perf["observations"]),
                "top_patterns": sorted(perf["response_patterns"].items(), 
                                      key=lambda x: x[1], reverse=True)[:3],
                "strengths": self._identify_model_strengths(model_id)
            }
        
        # Create overall ranking based on observations and pattern diversity
        rankings = []
        for model_id in valid_models:
            perf = self.model_performance[model_id]
            score = (
                perf["observations"] * 0.5 +
                len(perf["response_patterns"]) * 0.3 +
                (perf["context_usage"] / max(1, perf["observations"])) * 10
            )
            rankings.append((model_id, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        results["overall_ranking"] = rankings
        
        # Identify unique strengths
        for model_id in valid_models:
            strengths = results["model_comparisons"][model_id]["strengths"]
            for strength in strengths:
                is_unique = True
                for other_id in valid_models:
                    if other_id != model_id:
                        if strength in results["model_comparisons"][other_id]["strengths"]:
                            is_unique = False
                            break
                
                if is_unique:
                    if model_id not in results["unique_strengths"]:
                        results["unique_strengths"][model_id] = []
                    results["unique_strengths"][model_id].append(strength)
        
        return results
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get the current state of the learning engine"""
        return {
            "observation_count": self.observation_count,
            "learning_level": self.current_learning_level,
            "next_threshold": self.learning_thresholds[self.current_learning_level] 
                              if self.current_learning_level < len(self.learning_thresholds) else "max",
            "observed_models": list(self.model_performance.keys()),
            "learning_factors": self.learning_factors
        }
    
    def _extract_learning_features(self, 
                                  query: str, 
                                  response: str, 
                                  context: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extract learning features from an observed interaction
        """
        features = {
            "response_length": len(response),
            "sentence_count": response.count('.') + response.count('!') + response.count('?'),
            "paragraph_structure": response.count('\n\n'),
            "identified_patterns": self._identify_response_patterns(response),
            "context_references": self._count_context_references(response, context)
        }
        return features
    
    def _identify_response_patterns(self, response: str) -> List[str]:
        """Identify patterns in the response structure"""
        patterns = []
        
        # Check for opening patterns
        if response.startswith("I "):
            patterns.append("first_person_opening")
        elif any(response.lower().startswith(greeting) for greeting in ["hi ", "hello ", "greetings", "hey "]):
            patterns.append("greeting_opening")
        elif response.startswith("The "):
            patterns.append("factual_opening")
        
        # Check for structural patterns
        if ":\n" in response:
            patterns.append("list_format")
        if "```" in response:
            patterns.append("code_block")
        if "**" in response or "*" in response:
            patterns.append("emphasis_formatting")
        if response.count('\n\n') > 2:
            patterns.append("multi_paragraph")
        
        # Check for closing patterns
        if "?" in response[-5:]:
            patterns.append("question_closing")
        elif any(response.lower().endswith(closing) for closing in ["help!", "help.", "know.", "questions?"]):
            patterns.append("supportive_closing")
            
        return patterns
    
    def _count_context_references(self, 
                                 response: str, 
                                 context: Optional[List[Dict[str, Any]]]) -> int:
        """Count references to provided context in the response"""
        if not context:
            return 0
            
        reference_count = 0
        
        # Look for context terms in the response
        for ctx_item in context:
            if 'text' in ctx_item:
                # Get key terms from context
                context_text = ctx_item['text']
                key_terms = []
                
                # Extract potential key terms (simplified)
                words = context_text.split()
                for word in words:
                    # Only consider meaningful words of sufficient length
                    if len(word) > 5 and not word.lower() in ["there", "their", "would", "should", "could"]:
                        key_terms.append(word.lower())
                
                # Count matches in response
                for term in key_terms:
                    if term in response.lower():
                        reference_count += 1
        
        return min(10, reference_count)  # Cap at 10 to avoid skewing
    
    def _identify_model_strengths(self, model_id: str) -> List[str]:
        """Identify key strengths of a model based on observations"""
        strengths = []
        perf = self.model_performance[model_id]
        
        # Check for conciseness
        if perf["avg_response_length"] < 200 and perf["observations"] > 5:
            strengths.append("concise_responses")
        
        # Check for detail
        if perf["avg_response_length"] > 500 and perf["observations"] > 5:
            strengths.append("detailed_responses")
        
        # Check for context usage
        if (perf["context_usage"] / max(1, perf["observations"])) > 0.7:
            strengths.append("strong_context_integration")
        
        # Check for formatting
        has_formatting = False
        for pattern in perf["response_patterns"]:
            if pattern in ["list_format", "code_block", "emphasis_formatting"]:
                has_formatting = True
                break
        
        if has_formatting:
            strengths.append("effective_formatting")
        
        # Check for engagement patterns
        has_engagement = False
        for pattern in perf["response_patterns"]:
            if pattern in ["question_closing", "supportive_closing"]:
                has_engagement = True
                break
                
        if has_engagement:
            strengths.append("user_engagement")
        
        return strengths
    
    def _check_learning_progression(self) -> None:
        """Check if we've reached a learning threshold and update accordingly"""
        if (self.current_learning_level < len(self.learning_thresholds) and 
            self.observation_count >= self.learning_thresholds[self.current_learning_level]):
            
            # Upgrade learning level
            self.current_learning_level += 1
            
            # Enhance learning factors at each level
            for factor in self.learning_factors:
                self.learning_factors[factor] = min(0.95, self.learning_factors[factor] + 0.05)
                
            print(f"Nyptho learning progressed to level {self.current_learning_level}")
            print(f"Updated learning factors: {self.learning_factors}")
