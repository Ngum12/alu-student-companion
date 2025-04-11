from typing import List, Dict, Any, Optional
import re
import time

class BrainSearchEngine:
    """Handles search operations within the ALU Brain knowledge base"""
    
    def __init__(self):
        self._search_cache = {}  # Cache to improve performance
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "processing_time": []
        }
        print("Enhanced BrainSearchEngine initialized with caching and advanced semantic matching")
    
    def search(self, query: str, knowledge_base: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search in the ALU Brain knowledge base
        Uses advanced multi-layered scoring system with contextual relevance
        """
        # Add this at the beginning of the search method
        print(f"DEBUG: Search query: '{query}'")
        
        # Track statistics
        self._search_stats["total_searches"] += 1
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{query}:{top_k}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            self._search_stats["cache_hits"] += 1
            return cached_result
        
        results = []
        query_terms = self._preprocess_query(query)
        
        # Enhanced query classification
        query_intent = self._classify_query_intent(query)
        query_topics = self._extract_query_topics(query)
        
        # Search through all entries in all categories with weighted scoring
        for category, data in knowledge_base.items():
            # Calculate category relevance with topic matching
            category_relevance = self._calculate_category_relevance(category, query_terms, query_topics)
            
            for entry in data.get('entries', []):
                # Initialize comprehensive scoring system
                score = {
                    "category_match": category_relevance,
                    "question_match": 0.0,
                    "answer_match": 0.0,
                    "metadata_match": 0.0,
                    "type_match": 0.0,
                    "intent_match": 0.0,
                    "exact_match": 0.0
                }
                
                # Question matching with contextual weighting
                question = entry.get('question', '').lower()
                score["question_match"] = self._calculate_text_match_score(question, query_terms, weight=3)
                
                # Check for exact matches (highest priority)
                if question and query.lower() in question:
                    score["exact_match"] = 10.0
                
                # Answer content matching with semantic relevance
                answer = entry.get('answer', '').lower()
                score["answer_match"] = self._calculate_text_match_score(answer, query_terms, weight=1)
                
                # Metadata matching for additional context
                if 'metadata' in entry:
                    metadata_text = ' '.join([str(v) for v in entry['metadata'].values()]).lower()
                    score["metadata_match"] = self._calculate_text_match_score(metadata_text, query_terms, weight=0.5)
                
                # Entry type relevance based on query intent
                entry_type = entry.get('type', 'text')
                if self._is_type_relevant(entry_type, query_intent):
                    score["type_match"] = 2.0
                
                # Intent matching for better contextual relevance
                if query_intent and 'intent' in entry:
                    if entry['intent'] == query_intent:
                        score["intent_match"] = 3.0
                
                # Calculate final score as weighted sum
                final_score = (
                    score["category_match"] +
                    score["question_match"] +
                    score["answer_match"] +
                    score["metadata_match"] +
                    score["type_match"] +
                    score["intent_match"] +
                    score["exact_match"]
                )
                
                if final_score > 0:
                    results.append({
                        'entry': entry,
                        'category': category,
                        'score': final_score,
                        'score_breakdown': score
                    })
        
        # Sort by score and limit to top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:top_k]
        
        # Add to cache
        self._store_in_cache(cache_key, top_results)
        
        # Add this before returning results
        if top_results:
            print(f"DEBUG: Top result: {top_results[0]['category']} - {top_results[0]['entry'].get('question', 'No question')}")
            print(f"DEBUG: Score breakdown: {top_results[0]['score_breakdown']}")
        
        # Track processing time
        end_time = time.time()
        self._search_stats["processing_time"].append(end_time - start_time)
        
        # At the end of the search method, before returning top_results
        if not results:
            print("DEBUG: No results found for query!")
            # Check if there's code here that adds default entrepreneurship content
        
        return top_results  # Return empty list if no results found
    
    def _preprocess_query(self, query: str) -> List[str]:
        """Process the query to extract meaningful terms"""
        # Remove punctuation and convert to lowercase
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        
        # Split into terms and filter out common stop words and short terms
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'against', 'between',
                     'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up',
                     'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                     'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i',
                     'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours'}
        
        terms = [term for term in query.split() if term not in stop_words and len(term) > 2]
        
        # Add this debug output
        if not terms:
            print(f"DEBUG: Query '{query}' produced no search terms after preprocessing!")
        
        # Check for important terms that might be filtered
        important_terms = ['fee', 'due', 'pay', 'gpa', 'job', 'tax', 'aid', 'lab', 'web']
        for term in query.split():
            if term in important_terms and term not in terms:
                terms.append(term)
        
        return terms
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify the query intent for better matching"""
        query_lower = query.lower()
        
        # Define intent patterns
        intent_patterns = {
            'procedural': ['how to', 'how do i', 'process', 'steps', 'procedure', 'guide', 'instructions'],
            'informational': ['what is', 'what are', 'who is', 'explain', 'describe', 'tell me about'],
            'comparison': ['compare', 'difference', 'versus', 'vs', 'better', 'between'],
            'deadline': ['when', 'date', 'deadline', 'due', 'schedule', 'calendar', 'timeline'],
            'location': ['where', 'location', 'place', 'building', 'room', 'campus'],
            'contact': ['contact', 'email', 'phone', 'reach', 'speak', 'call'],
            'requirement': ['require', 'need', 'necessary', 'must have', 'should', 'mandatory']
        }
        
        # Check for matching patterns
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        
        return 'general'
    
    def _extract_query_topics(self, query: str) -> List[str]:
        """Extract topic areas from the query"""
        query_lower = query.lower()
        
        # Define topic keywords
        topics = {
            'academic': ['course', 'class', 'degree', 'major', 'minor', 'study', 'academic', 'grade', 'credit', 'transcript'],
            'admission': ['apply', 'admission', 'application', 'accept', 'reject', 'enroll'],
            'financial': ['tuition', 'fee', 'cost', 'payment', 'financial', 'aid', 'scholarship', 'loan', 'budget', 'fund'],
            'housing': ['dorm', 'housing', 'residence', 'apartment', 'live', 'roommate', 'accommodation'],
            'career': ['job', 'career', 'internship', 'employment', 'resume', 'interview', 'hire', 'company'],
            'administrative': ['office', 'staff', 'administration', 'policy', 'rule', 'regulation', 'requirement'],
            'student_life': ['club', 'organization', 'activity', 'event', 'social', 'community', 'student life'],
            'technology': ['computer', 'laptop', 'software', 'internet', 'wifi', 'technology', 'online', 'access'],
            'health': ['health', 'medical', 'doctor', 'nurse', 'counselor', 'wellness', 'sick', 'illness'],
            'international': ['visa', 'international', 'country', 'passport', 'foreign']
        }
        
        # Extract matching topics
        matching_topics = []
        for topic, keywords in topics.items():
            if any(keyword in query_lower for keyword in keywords):
                matching_topics.append(topic)
        
        return matching_topics if matching_topics else ['general']
    
    def _calculate_text_match_score(self, text: str, query_terms: List[str], weight: float = 1.0) -> float:
        """Calculate a weighted score for term matches in text"""
        if not text or not query_terms:
            return 0
            
        score = 0
        # Score exact phrase match higher
        if all(term in text for term in query_terms):
            score += 5 * weight
        
        # Score for individual terms with position weighting
        for term in query_terms:
            if term in text:
                # Term frequency
                term_count = text.count(term)
                score += min(term_count, 3) * weight  # Cap at 3 to avoid over-counting
                
                # Position weighting (terms at beginning are more important)
                position_idx = text.find(term)
                if position_idx <= len(text) * 0.2:  # First 20% of text
                    score += 2 * weight
                elif position_idx <= len(text) * 0.5:  # First half of text
                    score += 1 * weight
                
                # Exact word match (with word boundaries)
                word_pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(word_pattern, text):
                    score += 1.5 * weight
        
        # Add minimum score threshold to prevent weak matches
        if score < 0.5 * weight:  # Require a minimum relevant score
            return 0
        return score
    
    def _calculate_category_relevance(self, category: str, query_terms: List[str], query_topics: List[str]) -> float:
        """Determine how relevant a category is to the query based on terms and topics"""
        category_words = category.lower().replace('_', ' ').split()
        score = 0
        
        # Check for direct category matches in query terms
        common_terms = set(category_words) & set(query_terms)
        if common_terms:
            score += len(common_terms) * 2
        
        # Map category to topics
        category_topic_map = {
            'academic_policies': ['academic', 'administrative'],
            'academic_programs': ['academic'],
            'admissions': ['admission'],
            'campus_life': ['student_life', 'housing'],
            'career_services': ['career'],
            'financial_information': ['financial'],
            'student_resources': ['student_life', 'academic', 'technology', 'health']
        }
        
        # Add points for topic matches
        if category in category_topic_map:
            matching_topics = set(category_topic_map[category]) & set(query_topics)
            score += len(matching_topics) * 3
        
        return score
    
    def _is_type_relevant(self, entry_type: str, query_intent: str) -> bool:
        """Check if the entry type is relevant to the query intent"""
        # Map entry types to query intents
        type_intent_map = {
            'link_response': ['informational', 'resource'],
            'table_response': ['comparison', 'informational'],
            'statistical_response': ['informational', 'comparison'],
            'date_response': ['deadline', 'procedural'],
            'procedural_response': ['procedural', 'how_to'],
            'long_response': ['informational', 'explanation'],
            'short_response': ['general']
        }
        
        if entry_type in type_intent_map:
            return query_intent in type_intent_map[entry_type]
        
        return False
    
    def get_entry_by_id(self, entry_id: str, knowledge_base: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a specific entry by its ID"""
        for category, data in knowledge_base.items():
            for entry in data.get('entries', []):
                if entry.get('id') == entry_id:
                    return {
                        'entry': entry,
                        'category': category,
                        'score': 1.0
                    }
        return None
    
    def get_entries_by_category(self, category: str, knowledge_base: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get entries from a specific category with enhanced sorting"""
        if category not in knowledge_base:
            return []
            
        entries = knowledge_base[category].get('entries', [])
        
        # Sort entries by relevance (assuming more detailed entries are more important)
        sorted_entries = sorted(entries, 
                               key=lambda x: len(x.get('answer', '')) + len(str(x.get('metadata', {}))), 
                               reverse=True)
        
        return [{'entry': entry, 'category': category, 'score': 1.0} for entry in sorted_entries[:limit]]
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about search performance"""
        stats = self._search_stats.copy()
        
        # Calculate averages
        if stats["processing_time"]:
            stats["avg_processing_time"] = sum(stats["processing_time"]) / len(stats["processing_time"])
        else:
            stats["avg_processing_time"] = 0
            
        if stats["total_searches"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_searches"]
        else:
            stats["cache_hit_rate"] = 0
            
        return stats
    
    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get results from cache if not expired"""
        if key in self._search_cache:
            timestamp, results = self._search_cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return results
        return None
    
    def _store_in_cache(self, key: str, results: List[Dict[str, Any]]) -> None:
        """Store results in cache with timestamp"""
        self._search_cache[key] = (time.time(), results)
        
        # Clean up old cache entries
        self._clean_cache()
    
    def _clean_cache(self) -> None:
        """Remove expired cache entries"""
        # Clear the entire cache for now to rule out cache issues
        self._search_cache = {}
        
        # Alternatively, use the original implementation:
        # if len(self._search_cache) > 100:  # Only clean if cache is large
        #     now = time.time()
        #     expired_keys = [k for k, v in self._search_cache.items() if now - v[0] > self._cache_ttl]
        #     for k in expired_keys:
        #         del self._search_cache[k]
