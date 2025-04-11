from typing import Dict, Any
from duckduckgo_search import DDGS

# Update the search_web function to utilize conversation history
def search_web(query: str, conversation_history=None) -> Dict[str, Any]:
    """
    Search the web for answers to general knowledge questions.
    
    Args:
        query: The search query
        conversation_history: Optional previous conversation for context
        
    Returns:
        Dict with answer, snippets, and links
    """
    # If we have conversation history, use it to enhance the query
    enhanced_query = query
    if conversation_history and len(conversation_history) > 0:
        # Get the last few messages to provide context
        recent_messages = conversation_history[-3:]  # Last 3 messages
        
        # Check if the query could be referring to previous conversation
        if any(word in query.lower() for word in ["it", "that", "this", "they", "those", "them", "he", "she"]):
            # Create context from recent messages
            context = " ".join([msg["content"] for msg in recent_messages])
            enhanced_query = f"{context} {query}"
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(enhanced_query, max_results=5))
            
        if not results:
            return {
                "answer": f"I couldn't find information about '{query}' online.",
                "snippets": [],
                "links": []
            }
            
        # Extract snippets and links
        snippets = [result.get("body", "") for result in results]
        links = [result.get("href", "") for result in results]
        titles = [result.get("title", "") for result in results]
        
        # Prepare the answer
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
            "answer": "I encountered an error while searching for information online.",
            "snippets": [],
            "links": []
        }