from duckduckgo_search import DDGS
from typing import Dict, List, Any
import re

def is_general_knowledge_question(question: str) -> bool:
    """Determine if a question is likely a general knowledge question."""
    # Keywords that suggest general knowledge questions
    knowledge_keywords = [
        "who", "what", "when", "where", "why", "how", "did", "does", "do", 
        "is", "are", "was", "were", "will", "can", "could", "should", "would",
        "history", "world", "country", "president", "minister", "capital", "population",
        "fact", "information", "leader", "government", "director"
    ]
    
    question_lower = question.lower()
    
    # Check for presence of knowledge keywords
    for keyword in knowledge_keywords:
        if keyword in question_lower:  # This is more lenient
            return True
    
    return False

def search_web(query: str, conversation_history=None) -> Dict[str, Any]:
    """
    Search the web for answers to general knowledge questions.
    
    Args:
        query: The search query
        conversation_history: Optional previous conversation for context
        
    Returns:
        Dict with answer, snippets, and links
    """
    # Initialize enhanced_query with the original query
    enhanced_query = query
    
    # If we have conversation history, use it to enhance the query
    if conversation_history and len(conversation_history) > 0:
        # Get just the last user message for context (not the AI response)
        recent_user_messages = [msg for msg in conversation_history[-4:] if msg.get("role") == "user"]
        
        # Check if the query could be referring to previous conversation
        if any(word in query.lower() for word in ["it", "that", "this", "they", "those", "them", "he", "she", "steps", "instructions"]):
            # Create context from recent messages - but keep it brief!
            if recent_user_messages:
                # Just use the most recent user query, not the entire conversation
                context = recent_user_messages[-1]["content"]
                # Limit context length to avoid overly long queries
                if len(context) > 100:
                    context = context[:100]
                enhanced_query = f"{context} {query}"
                print(f"Enhanced query: '{enhanced_query}'")
    
    try:
        print(f"Searching web for: '{enhanced_query}'")
        with DDGS() as ddgs:
            # Limit query length to avoid API issues
            if len(enhanced_query) > 200:
                enhanced_query = enhanced_query[:200]
            results = list(ddgs.text(enhanced_query, max_results=5))
            
        if not results:
            return {
                "answer": f"I couldn't find information about '{query}' online.",
                "snippets": [],
                "links": [],
                "titles": []
            }
            
        # Extract snippets and links
        snippets = [result.get("body", "") for result in results]
        links = [result.get("href", "") for result in results]
        titles = [result.get("title", "") for result in results]
        
        # Prepare the answer - include specific steps when requested
        if "how" in query.lower() or "steps" in query.lower() or "instructions" in query.lower():
            answer = f"Here are the steps based on web search results:\n\n"
            
            # Extract step-by-step information from snippets
            for snippet in snippets[:2]:
                # Look for numbered steps or bullet points
                step_matches = re.findall(r'(\d+\.\s*[^\.\n]+)', snippet)
                if step_matches:
                    for step in step_matches[:5]:  # Limit to 5 steps
                        answer += f"{step.strip()}\n"
                    break
                
                # If no numbered steps, just include the first snippet with more detail
                else:
                    answer += f"{snippet}\n\n"
        else:
            answer = f"Based on web search results:\n\n"
            for i, snippet in enumerate(snippets[:2]):
                # Add the most relevant snippets
                answer += f"{snippet[:300]}...\n\n"
        
        return {
            "answer": answer,
            "snippets": snippets,
            "links": links,
            "titles": titles
        }
    except Exception as e:
        print(f"Error in web search: {e}")
        return {
            "answer": f"I encountered an error while searching for information online: {str(e)}",
            "snippets": [],
            "links": [],
            "titles": []
        }