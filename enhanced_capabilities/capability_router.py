import re
from typing import Dict, Any, List, Callable

# Fix the imports to match the implementations
from .math_solver import is_math_question, solve_math_problem
from .web_lookup import is_general_knowledge_question, search_web
from .code_support import is_code_question, handle_code_question

def format_math_solution(answer, steps):
    """
    Format the math solution with its steps for better display.
    
    Args:
        answer: The final answer to the math problem
        steps: List of steps taken to solve the problem
        
    Returns:
        A formatted string with the solution and steps
    """
    # Format the main answer with bold styling and clear separation
    formatted_solution = f"### {answer}\n\n"
    
    # Add a heading for the solution steps
    formatted_solution += "**Solution Steps:**\n\n"
    
    # Format each step with proper spacing and numbering where appropriate
    step_number = 1
    for step in steps:
        # Skip the initial repeat of the question
        if step.startswith("Starting with"):
            formatted_solution += f"1️⃣ {step}\n\n"
            step_number = 2
            continue
            
        # Format verification section specially
        if step.startswith("Verification"):
            formatted_solution += f"\n**Verification:**\n"
            formatted_solution += f"- {step.replace('Verification: ', '')}\n"
            continue
            
        # Format left side/right side specially for alignment
        if step.startswith("Left side:"):
            formatted_solution += f"- {step}\n"
            continue
            
        if step.startswith("Right side:"):
            formatted_solution += f"- {step}\n"
            continue
            
        if step.startswith("Since"):
            formatted_solution += f"- {step}\n\n"
            continue
            
        # Format equations in code blocks for better readability
        if "=" in step and not step.startswith("Solution") and not step.startswith("Step"):
            formatted_solution += f"```\n{step}\n```\n\n"
            continue
            
        # Format regular solution steps
        if step.startswith("Step"):
            number_emoji = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"][step_number-1] if step_number <= 10 else f"{step_number}."
            formatted_solution += f"{number_emoji} {step.replace('Step ' + str(step_number-1) + ':', '')}\n\n"
            step_number += 1
            continue
            
        # Default formatting for other steps
        formatted_solution += f"{step}\n\n"
    
    return formatted_solution

def is_school_related(question: str) -> bool:
    """
    Determine if a question is related to school/academic matters.
    
    Args:
        question: The user's question
        
    Returns:
        True if it appears to be school-related, False otherwise
    """
    # Keywords that suggest school-related questions
    school_keywords = [
        "alu", "african leadership university", "campus", "course", "class", "professor",
        "teacher", "lecturer", "student", "degree", "major", "minor", "graduation",
        "academic", "semester", "term", "grade", "exam", "test", "assignment",
        "homework", "deadline", "syllabus", "tuition", "scholarship", "financial aid",
        "dormitory", "residence hall", "library", "textbook", "schedule"
    ]
    
    question_lower = question.lower()
    
    # Check for the presence of school-related keywords
    for keyword in school_keywords:
        if keyword in question_lower:
            return True
    
    return False

def handle_question(query: str) -> Dict[str, Any]:
    """Route questions to appropriate capability"""
    
    # Check for greetings/farewells FIRST before other processing
    query_lower = query.lower().strip()
    
    # Detect greetings
    if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "greetings"]):
        return {
            "capability": "greeting",
            "answer": "Hello! I'm the ALU Assistant. How can I help with your questions today?",
            "source": "conversation"
        }
    
    # Detect farewells
    if any(farewell in query_lower for farewell in ["bye", "goodbye", "see you", "farewell"]):
        return {
            "capability": "farewell",
            "answer": "Goodbye! Feel free to return if you have more questions about ALU. Wishing you success in your studies!",
            "source": "conversation"
        }
        
    # Rest of your existing routing logic...
    
    print(f"Routing question: '{query}'")
    
    # 1. First, check if it's a greeting/simple message (don't route these)
    greeting_patterns = [
        r"^hi\b", r"^hello\b", r"^hey\b", r"^good\s+(morning|afternoon|evening|day)",
        r"^thanks", r"^thank\s+you", r"^ok\b", r"^okay\b", r"^\s*$"
    ]
    
    for pattern in greeting_patterns:
        if re.search(pattern, query.lower()):
            print("Detected as greeting/simple message")
            return {
                "answer": "Hello! How can I help you today?",
                "source": "greeting",
                "additional_info": {}
            }
    
    # 2. Check if it's geographical/knowledge question (should be handled by web search)
    geo_patterns = [
        r"capital\s+of", r"where\s+is", r"location\s+of", r"country", 
        r"city", r"continent", r"population\s+of", r"president\s+of"
    ]
    
    for pattern in geo_patterns:
        if re.search(pattern, query.lower()):
            print("Detected as geographical/knowledge question")
            try:
                search_result = search_web(query, None)
                return {
                    "answer": search_result.get("answer", "I couldn't find information about this."),
                    "source": "web_search",
                    "additional_info": {
                        "snippets": search_result.get("snippets", []),
                        "links": search_result.get("links", [])
                    }
                }
            except Exception as e:
                print(f"Web search error: {e}")
                # Fall through to other options
    
    # 3. Try code support
    if is_code_question(query):
        try:
            result = handle_code_question(query)
            return {
                "answer": result.get("answer", "I couldn't analyze this code."),
                "source": "code_support",
                "additional_info": result
            }
        except Exception as e:
            print(f"Code support error: {e}")
    
    # 4. Try math solving (make sure the question actually has numbers or math symbols)
    if is_math_question(query) and re.search(r'[0-9+\-*/^=]', query):
        try:
            answer, steps = solve_math_problem(query)
            # Use the new formatter for better math display
            response = format_math_solution(answer, steps)
            return {
                "answer": response,
                "source": "math_solver",
                "additional_info": steps
            }
        except Exception as e:
            print(f"Math solver error: {e}")
    
    # 5. Try web search for other general knowledge (most flexible)
    if is_general_knowledge_question(query):
        try:
            # Pass conversation history to web search
            search_result = search_web(query, None)
            return {
                "answer": search_result.get("answer", "I couldn't find information about this."),
                "source": "web_search",
                "additional_info": {
                    "snippets": search_result.get("snippets", []),
                    "links": search_result.get("links", [])
                }
            }
        except Exception as e:
            print(f"Web search error: {e}")
    
    # 6. If we got here, use the provided search function if available
    if None:
        try:
            docs = None(query)
            # Process docs here if needed
            return {
                "answer": "I found some information in our knowledge base that might help.",
                "source": "document_search",
                "additional_info": {"docs": docs}
            }
        except Exception as e:
            print(f"Document search error: {e}")
    
    # 7. Default fallback
    return {
        "answer": "I don't have enough information to answer that question.",
        "source": "fallback",
        "additional_info": {}
    }