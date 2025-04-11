import random
import time
import re
from typing import List, Dict, Any, Optional
import markdown

from retrieval_engine import Document

class ResponseGenerator:
    """Handles the generation of responses based on context and query"""
    
    def __init__(self):
        self.response_cache = {}
        self.cache_ttl = 300  # 5 minutes
        print("Enhanced ResponseGenerator initialized with caching and advanced formatting")
    
    def generate_response(self, 
                         query: str, 
                         context: List[Document], 
                         role: str = "student") -> str:
        """
        Generate a structured, professional response using context and role
        with proper markdown formatting and knowledge integration
        """
        # Check cache for this query
        cache_key = f"{query}:{role}:{len(context)}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Start crafting response
        response_parts = []
        
        # Add appropriate greeting
        greeting = self._create_greeting(query, role)
        if greeting:
            response_parts.append(greeting)
        
        # Process context from ALU Brain
        brain_content = self._process_brain_content(query, context)
        if brain_content:
            response_parts.append(brain_content)
        else:
            # If no ALU Brain content, generate a response using other context
            general_content = self._generate_general_response(query, context, role)
            response_parts.append(general_content)
        
        # Add appropriate closing
        closing = self._create_closing(query, role)
        if closing:
            response_parts.append(closing)
        
        # Format as markdown
        full_response = "\n\n".join(response_parts)
        
        # Add to cache
        self._add_to_cache(cache_key, full_response)
        
        return full_response
    
    def _process_brain_content(self, query: str, context: List[Document]) -> str:
        """Process ALU Brain content in the context to create a structured response"""
        # Identify ALU Brain documents
        alu_brain_docs = [doc for doc in context if "ALU Brain" in doc.metadata.get('source', '')]
        
        if not alu_brain_docs:
            return ""
            
        # Choose the most relevant document(s)
        # Sort by score if available
        alu_brain_docs.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        
        # Add this after sorting the documents
        if alu_brain_docs and hasattr(alu_brain_docs[0], 'score') and alu_brain_docs[0].score is not None:
            top_score = alu_brain_docs[0].score
        else:
            top_score = 0.5  # Default score when none available
            
        min_acceptable_score = max(0.5, top_score * 0.7)  # At least 70% as relevant as top doc

        response_parts = []
        used_categories = set()
        doc_count = 0
        
        for doc in alu_brain_docs:
            doc_score = getattr(doc, 'score', None)
            if doc_score is None or doc_score < min_acceptable_score:
                continue  # Skip documents that aren't relevant enough
                
            # Limit to max 3 documents for conciseness
            if doc_count >= 3:
                break
                
            source_category = doc.metadata.get('source', '').replace('ALU Brain: ', '')
            
            # Skip if we already used this category
            if source_category in used_categories:
                continue
                
            used_categories.add(source_category)
            doc_count += 1
            
            # Format based on document type
            doc_type = doc.metadata.get('type', 'text')
            
            if 'link' in doc_type:
                response_parts.append(self._format_link_response(doc, source_category))
            elif 'table' in doc_type or 'statistical' in doc_type:
                response_parts.append(self._format_data_response(doc, source_category))
            elif 'procedural' in doc_type:
                response_parts.append(self._format_procedural_response(doc, source_category))
            else:
                response_parts.append(self._format_text_response(doc, source_category))
        
        if response_parts:
            return "\n\n".join(response_parts)
        else:
            return ""
    
    def _format_link_response(self, doc: Document, category: str) -> str:
        """Format a link response with proper markdown"""
        title = doc.metadata.get('title', 'Resources')
        text = doc.text
        
        # Extract links using regex
        links = re.findall(r'- (.+?): (https?://\S+)', text)
        
        response = f"## {title}\n\n"
        
        # Extract the main content (question and answer)
        main_content = re.split(r'\n\n- ', text, 1)[0]
        response += f"{main_content}\n\n"
        
        if links:
            response += "### Relevant Resources\n\n"
            for link_name, link_url in links:
                response += f"* [{link_name}]({link_url})\n"
        
        response += f"\n*Source: ALU {category.replace('_', ' ').title()}*"
        return response
    
    def _format_data_response(self, doc: Document, category: str) -> str:
        """Format a data/statistical response with proper markdown tables"""
        title = doc.metadata.get('title', 'Information')
        text = doc.text
        
        response = f"## {title}\n\n"
        
        # Split into sections
        parts = text.split("\n\n")
        
        # Add main text
        if parts and len(parts) > 1:
            response += f"{parts[0]}\n\n{parts[1]}\n\n"
        else:
            response += f"{text}\n\n"
        
        # Look for table data
        table_match = re.search(r"Table data:\n([\s\S]+)", text)
        if table_match:
            table_text = table_match.group(1)
            rows = table_text.strip().split("\n")
            
            # Create markdown table
            if rows:
                headers = rows[0].strip().replace("  ", "").split(", ")
                response += "| " + " | ".join(headers) + " |\n"
                response += "| " + " | ".join(["---" for _ in headers]) + " |\n"
                
                for row in rows[1:]:
                    cells = row.strip().replace("  ", "").split(", ")
                    response += "| " + " | ".join(cells) + " |\n"
        
        response += f"\n*Source: ALU {category.replace('_', ' ').title()}*"
        return response
    
    def _format_procedural_response(self, doc: Document, category: str) -> str:
        """Format a procedural response with numbered steps"""
        title = doc.metadata.get('title', 'Process')
        text = doc.text
        
        response = f"## {title}\n\n"
        
        # Extract the main content and steps
        parts = text.split("\n\n")
        if len(parts) >= 2:
            response += f"{parts[0]}\n\n{parts[1]}\n\n"
        else:
            response += f"{parts[0]}\n\n"
        
        # Extract numbered steps if present
        steps_match = re.search(r"\n(\d+\..+(?:\n\d+\..+)*)", text)
        if steps_match:
            steps = steps_match.group(1)
            response += "### Steps\n\n" + steps + "\n\n"
        
        response += f"\n*Source: ALU {category.replace('_', ' ').title()}*"
        return response
    
    def _format_text_response(self, doc: Document, category: str) -> str:
        """Format a general text response"""
        title = doc.metadata.get('title', 'Information')
        text = doc.text
        
        # Split the text by question/answer if possible
        parts = text.split("\n\n", 1)
        
        response = f"## {title}\n\n"
        
        if len(parts) > 1:
            answer = parts[1]
            response += f"{answer}\n\n"
        else:
            response += f"{text}\n\n"
        
        response += f"\n*Source: ALU {category.replace('_', ' ').title()}*"
        return response
    
    def _generate_general_response(self, query: str, context: List[Document], role: str) -> str:
        """Generate a response using non-ALU Brain context with a friendly tone"""
        query_lower = query.lower()
        response = ""
        
        # Try to extract useful information from context
        if context:
            # Get the most relevant document
            context.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
            best_doc = context[0]
            
            # REMOVED: No more "About Your Query" heading
            # Instead, use a friendly opener based on the query content
            
            # Extract most relevant content
            content = best_doc.text[:400]  # Limit length
            
            # Create friendly opening based on query content
            if "graduation" in query_lower:
                opener = "🎓 Congratulations on your journey toward graduation! "
            elif "course" in query_lower or "class" in query_lower:
                opener = "📚 Ready to dive into some exciting learning? "
            elif "campus" in query_lower:
                opener = "🏫 Our beautiful campuses are waiting for you! "
            elif "scholarship" in query_lower or "financial" in query_lower:
                opener = "💰 Let's make your education more affordable! "
            elif "deadline" in query_lower:
                opener = "⏰ Don't worry, I've got the important dates for you! "
            elif "faculty" in query_lower or "professor" in query_lower:
                opener = "👨‍🏫 Our amazing faculty members are here to support you! "
            elif "help" in query_lower or "confused" in query_lower:
                opener = "🤗 We've all been there! Let me help clear things up: "
            elif "thank" in query_lower:
                opener = "😊 You're very welcome! Here's what you asked about: "
            else:
                # Use rotating friendly openers for general queries
                friendly_openers = [
                    "✨ Great question! ",
                    "🔍 I found just what you're looking for! ",
                    "💡 Here's something helpful: ",
                    "🌟 I'm happy to share this with you: ",
                    "🚀 Let's explore this together: "
                ]
                opener = random.choice(friendly_openers)
            
            response = f"{opener}{content}"
            
            if best_doc.metadata.get('source'):
                response += f"\n\n*Source: {best_doc.metadata.get('source')}*"
        else:
            # Create role-specific responses when no context is available
            # [Keep your existing role-specific responses but add friendly emojis]
            if role == "admin":
                response = "🔑 **Administrative Tools & Resources**\n\nAs an ALU administrator, you have access to various systems and tools to help manage university operations. If you need specific information about administrative procedures, please provide more details about your query."
            elif role == "faculty":
                response = "👨‍🏫 **Faculty Resources**\n\nAs a faculty member at ALU, you can access various teaching resources and student management tools. For more specific guidance on course materials or academic processes, please provide additional details about your requirements."
            else:  # student or default
                if "course" in query_lower or "class" in query_lower:
                    response = "📚 **ALU Courses**\n\nALU's curriculum is designed to be practical and focused on developing leadership skills. Courses integrate real-world challenges and emphasize both technical expertise and soft skills development. For specific course information, please mention the course name or code."
                elif "assignment" in query_lower or "homework" in query_lower:
                    response = "✍️ **Assignment Guidelines**\n\nALU assignments are designed to be practical and applicable to real-world situations. When working on assignments, make sure to:\n\n1. Follow the rubric carefully\n2. Connect concepts to real-world scenarios\n3. Demonstrate critical thinking\n4. Cite sources properly\n5. Submit before the deadline through the designated platform"
                elif "campus" in query_lower or "facility" in query_lower:
                    response = "🏫 **Campus Facilities**\n\nALU campuses feature state-of-the-art facilities designed to enhance the learning experience, including:\n\n* Collaborative learning spaces\n* Technology labs with modern equipment\n* Library and research resources\n* Student social areas\n* Quiet study zones\n* Sports and recreation facilities"
                else:
                    response = "🦁 **African Leadership University**\n\nALU is committed to developing the next generation of African leaders through innovative education approaches. Our programs focus on real-world challenges and developing both technical expertise and leadership capabilities.\n\nFor more specific information about programs, admissions, or campus life, please provide additional details about your area of interest."
        
        return response
    
    def _create_greeting(self, query: str, role: str) -> str:
        """Create an appropriate greeting based on the query and role"""
        # Determine if query is a question
        is_question = '?' in query
        
        if is_question:
            return "Thank you for your question. Here's what I found for you:"
        else:
            return "Here's some information based on your request:"
    
    def _create_closing(self, query: str, role: str) -> str:
        """Create a friendly, personalized closing based on the query and role"""
        query_lower = query.lower()
        
        # Different closings for different types of queries
        if "thank" in query_lower:
            closings = [
                "😊 You're very welcome! Always happy to help you succeed at ALU!",
                "🙌 Anytime! Your success is what we're here for!",
                "💫 It's my pleasure to help! Let me know if you need anything else!"
            ]
        elif "help" in query_lower or "problem" in query_lower:
            closings = [
                "🤝 I hope that helps! Remember, challenges are just opportunities in disguise!",
                "💪 You've got this! And the ALU community is here to support you every step of the way.",
                "🌈 Don't hesitate to reach out again if you need more guidance. We're in this together!"
            ]
        elif "graduation" in query_lower or "future" in query_lower:
            closings = [
                "🎓 Wishing you all the best on your journey to graduation and beyond!",
                "🚀 The future is bright for ALU graduates! Keep pushing forward!",
                "⭐ Your ALU journey is preparing you for amazing things ahead!"
            ]
        elif "deadline" in query_lower or "worried" in query_lower:
            closings = [
                "⏰ Don't worry! With good planning, you'll meet those deadlines successfully!",
                "📅 Take a deep breath - you've got this! One step at a time.",
                "🌟 Stay organized and focused. You're more capable than you realize!"
            ]
        else:
            closings = [
                "✨ Hope this helps! Feel free to ask if you have more questions!",
                "🌍 Making a difference in Africa starts with being well-informed. Glad I could help!",
                "💡 Anything else you're curious about? I'm here to support your ALU journey!",
                "🦁 The ALU pride is always here to help you succeed!",
                "🌟 Wishing you an amazing day filled with learning and growth!"
            ]
        
        return random.choice(closings)
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """Retrieve response from cache if not expired"""
        if key in self.response_cache:
            timestamp, response = self.response_cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return response
        return None
    
    def _add_to_cache(self, key: str, response: str) -> None:
        """Add response to cache with timestamp"""
        self.response_cache[key] = (time.time(), response)
        
        # Clean up old cache entries if cache gets too large
        if len(self.response_cache) > 100:
            now = time.time()
            expired_keys = [k for k, v in self.response_cache.items() 
                           if now - v[0] > self.cache_ttl]
            for k in expired_keys:
                del self.response_cache[k]
