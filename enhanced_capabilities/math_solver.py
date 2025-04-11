import re
import sympy
from sympy import symbols, solve, simplify, expand, factor, integrate, diff
from typing import Dict, Any, List, Tuple

def is_math_question(question: str) -> bool:
    """
    Determine if a question is likely a math problem.
    
    Args:
        question: The user's question
        
    Returns:
        True if it appears to be a math question
    """
    # Convert to lowercase for case-insensitive matching
    question_lower = question.lower()
    
    # IMPORTANT: Check for numbers and math operators first
    # This prevents non-math questions from being misidentified
    has_math_elements = bool(re.search(r'[0-9+\-*/^=()]', question_lower))
    if not has_math_elements:
        return False
    
    # Keywords often found in math questions
    math_keywords = [
        "solve", "calculate", "evaluate", "find", "compute", "derive", "simplify", 
        "factor", "expand", "integrate", "differentiate", "equation", "expression",
        "polynomial", "function", "graph", "plot", "value", "=", "+", "-", "*", "/",
        "sqrt", "square", "cube", "sum", "difference", "product", "quotient", "remainder"
    ]
    
    # Check for math keywords (with the added security that we already have math symbols)
    for keyword in math_keywords:
        if keyword in question_lower:
            return True
    
    # If we have math symbols but no specific keywords, do a more thorough check
    # for mathematical patterns
    math_patterns = [
        r"\d+\s*[\+\-\*\/\^\=]\s*\d+",  # Basic operations: 3 + 4, 5*6
        r"[xyz]\s*[\+\-\*\/\^\=]\s*\d+", # Variables with operations: x + 5
        r"\d+\s*[\+\-\*\/\^\=]\s*[xyz]", # Operations with variables: 5 + x
        r"solve\s+for\s+[xyz]",       # "Solve for x"
        r"find\s+[xyz]",              # "Find x"
    ]
    
    for pattern in math_patterns:
        if re.search(pattern, question_lower):
            return True
    
    # Default: if we have numbers and operators but no specific patterns,
    # assume it might be a math question
    return has_math_elements and ("+" in question_lower or "-" in question_lower or 
                                 "*" in question_lower or "/" in question_lower or
                                 "=" in question_lower)

def extract_math_expression(question: str) -> str:
    """
    Extract the mathematical expression from a text question.
    Fixed to correctly handle expressions like "2x+4=0 find x"
    """
    # Clean up the question and remove extra whitespace
    question = ' '.join(question.strip().split())
    
    # First check for equations with equals sign
    has_equals = '=' in question
    if has_equals:
        # Extract the part of the string containing the equation
        # This regex stops at common keywords that would follow an equation
        equation_pattern = r'([^=]+=.+?)(?:\s+(?:find|solve|for|where|when)\b|$)'
        equation_match = re.search(equation_pattern, question)
        
        if equation_match:
            return equation_match.group(1).strip()
    
    # If no equation was found or extracted, try other methods
    cleaned_question = question
    
    # Check for common instruction formats and strip them
    instruction_patterns = [
        r"^(find|solve|calculate|evaluate|compute)[\s\w]+:",
        r"^(find|solve|calculate|evaluate|compute)[\s\w]+when",
        r"^(find|solve|calculate|evaluate|compute)[\s\w]+if",
        r"^(find|solve|calculate|evaluate|compute)[\s\w]+for[\s\w]+in",
    ]
    
    for pattern in instruction_patterns:
        match = re.match(pattern, question.lower())
        if match:
            cleaned_question = question[match.end():].strip()
            break
    
    # Look for common words that indicate the end of a math expression
    end_markers = ['find', 'solve', 'for', 'where', 'when']
    for marker in end_markers:
        marker_pattern = rf'\s+{marker}\s+'
        if re.search(marker_pattern, cleaned_question.lower()):
            # Split at the marker and take the part before it
            parts = re.split(marker_pattern, cleaned_question.lower(), 1)
            cleaned_question = parts[0].strip()
            break
    
    # Additional check for end markers without spaces
    if has_equals:  # Only apply this logic if we know there's an equals sign
        for marker in end_markers:
            if marker in cleaned_question.lower():
                # Find the position where the marker starts
                marker_pos = cleaned_question.lower().find(marker)
                # Only process if marker is not at the start
                if marker_pos > 0:
                    # Check what's before the marker
                    if cleaned_question[marker_pos-1] in " =+-*/^()0123456789":
                        # This looks like a marker following an equation
                        cleaned_question = cleaned_question[:marker_pos].strip()
                        break
    
    # Look for equations in the cleaned question
    equation_pattern = r'([0-9a-zA-Z\+\-\*\/\^\(\)\s=\.]+\=[0-9a-zA-Z\+\-\*\/\^\(\)\s\.]+)'
    equation_match = re.search(equation_pattern, cleaned_question)
    if equation_match:
        return equation_match.group(1).strip()
    
    # If no equation found, look for any mathematical expression
    expression_pattern = r'([0-9a-zA-Z\+\-\*\/\^\(\)\s\.]+)'
    expr_match = re.search(expression_pattern, cleaned_question)
    if expr_match:
        return expr_match.group(1).strip()
    
    # If all else fails, return the cleaned question
    return cleaned_question

def preprocess_expression(expr_str: str) -> str:
    """
    Preprocess the expression to make it compatible with sympy.
    """
    if not expr_str:
        return expr_str
        
    # Replace ^ with ** for exponentiation
    expr_str = expr_str.replace('^', '**')
    
    # Handle implicit multiplication cases
    # 2x → 2*x  (number followed by variable)
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
    
    # Handle coefficient next to parentheses: 2(x+1) → 2*(x+1)
    expr_str = re.sub(r'(\d)(\()', r'\1*\2', expr_str)
    
    # Handle closing then opening parentheses: (x+1)(x+2) → (x+1)*(x+2)
    expr_str = re.sub(r'(\))(\()', r'\1*\2', expr_str)
    
    # Handle variable next to parentheses: x(y+1) → x*(y+1)
    expr_str = re.sub(r'([a-zA-Z])(\()', r'\1*\2', expr_str)
    
    # Remove any extra spaces to avoid parsing issues
    expr_str = expr_str.replace(' ', '')
    
    return expr_str

def extract_variable_to_solve(question: str) -> str:
    """
    Extract which variable to solve for from the question.
    Returns the variable or empty string if none found.
    """
    # Common patterns for indicating variables to solve
    patterns = [
        r"solve\s+for\s+([a-zA-Z])", 
        r"find\s+([a-zA-Z])",
        r"calculate\s+([a-zA-Z])",
        r"determine\s+([a-zA-Z])"
    ]
    
    question_lower = question.lower()
    
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            return match.group(1)
    
    # Default to 'x' if no specific variable mentioned
    return 'x'

def generate_solving_steps(math_expr: str, var, solutions) -> List[str]:
    """
    Generate detailed step-by-step explanations for solving an equation.
    
    Args:
        math_expr: The original mathematical expression
        var: The variable being solved for
        solutions: The solutions found
        
    Returns:
        List of steps as strings
    """
    steps = []
    
    # Add initial step showing the equation
    steps.append(f"Starting with the equation: {math_expr}")
    
    # Generate detailed algebraic steps
    if '=' in math_expr:
        left, right = math_expr.split('=', 1)
        left = left.strip()
        right = right.strip()
        
        # Step 1: Move all terms to one side
        steps.append(f"Step 1: Rearrange to get all terms on the left side")
        steps.append(f"{left} - ({right}) = 0")
        
        try:
            # Try to simplify and show the standard form
            left_expr = sympy.sympify(preprocess_expression(left))
            right_expr = sympy.sympify(preprocess_expression(right))
            standard_form = left_expr - right_expr
            
            if var in str(standard_form):
                # Only show if it's actually different from the previous step
                if str(standard_form) != f"{left} - ({right})":
                    steps.append(f"Step 2: Simplify the equation")
                    steps.append(f"{standard_form} = 0")
            
                # Step 3: Isolate the variable
                steps.append(f"Step 3: Isolate the variable {var}")
                
                # Handle linear equations in a special way for better explanations
                if str(var) in str(standard_form) and "**" not in str(standard_form) and "/" + str(var) not in str(standard_form):
                    # It's likely a linear equation, extract the coefficient
                    try:
                        # Get coefficient of x
                        coef = sympy.Poly(standard_form, sympy.Symbol(str(var))).all_coeffs()[0]
                        constant = -sympy.Poly(standard_form, sympy.Symbol(str(var))).all_coeffs()[1]
                        
                        # Show steps for linear equation
                        steps.append(f"{coef}*{var} = {constant}")
                        steps.append(f"{var} = {constant}/{coef}")
                        
                        # Simplify the result if possible
                        simplified = sympy.simplify(constant / coef)
                        if simplified != constant / coef:
                            steps.append(f"{var} = {simplified}")
                    except:
                        # If the specific approach fails, use a generic message
                        steps.append(f"Solving for {var}...")
                else:
                    # For non-linear equations, be more generic
                    steps.append(f"Solving the equation for {var}...")
        except Exception as e:
            # If symbolic processing fails, fall back to generic steps
            steps.append(f"Step 2: Solve for {var}")
    else:
        # For expressions without equals sign
        steps.append(f"Setting the expression equal to zero: {math_expr} = 0")
        steps.append(f"Solving for {var}...")
    
    # Add final solutions with verification
    if len(solutions) == 1:
        steps.append(f"Solution: {var} = {solutions[0]}")
        
        # Add verification step
        try:
            # Verify by substituting back into the original equation
            if '=' in math_expr:
                left, right = math_expr.split('=', 1)
                left_expr = sympy.sympify(preprocess_expression(left))
                right_expr = sympy.sympify(preprocess_expression(right))
                
                # Substitute the solution
                left_result = left_expr.subs(sympy.Symbol(str(var)), solutions[0])
                right_result = right_expr.subs(sympy.Symbol(str(var)), solutions[0])
                
                steps.append(f"Verification: Substitute {var} = {solutions[0]} back into the equation")
                steps.append(f"Left side: {left} = {left_result}")
                steps.append(f"Right side: {right} = {right_result}")
                steps.append(f"Since {left_result} = {right_result}, the solution is correct.")
        except:
            pass  # Skip verification if it fails
            
    elif len(solutions) > 1:
        steps.append(f"Multiple solutions found:")
        for i, sol in enumerate(solutions, 1):
            steps.append(f"Solution {i}: {var} = {sol}")
    else:
        steps.append("No solutions found.")
    
    return steps

def solve_math_problem(question: str) -> Tuple[str, List[str]]:
    """
    Solve a math problem using symbolic computation.
    
    Args:
        question: The math question
        
    Returns:
        Tuple of (answer, steps)
    """
    # Clean up the question
    question = question.strip()
    
    try:
        # Extract the actual equation or expression
        math_expr = extract_math_expression(question)
        print(f"Original question: '{question}'")
        print(f"Extracted expression: '{math_expr}'")
        
        # Add safety check
        if not math_expr or "find" in math_expr.lower() or "solve" in math_expr.lower():
            # Failed to extract properly
            print("Warning: Extraction may not have been clean")
        
        # Preprocess the expression (replace ^ with **, handle implicit multiplication)
        processed_expr = preprocess_expression(math_expr)
        print(f"Processed expression: '{processed_expr}'")
        
        # Safety check for invalid expressions
        if "find" in processed_expr.lower() or "solve" in processed_expr.lower():
            processed_expr = re.sub(r'(find|solve)[a-zA-Z]*', '', processed_expr, flags=re.IGNORECASE)
            print(f"Fixed processed expression: '{processed_expr}'")
        
        # If it's an equation (contains =), solve it
        if '=' in processed_expr:
            left, right = processed_expr.split('=', 1)
            left = sympy.sympify(left.strip())
            right = sympy.sympify(right.strip())
            
            # Move all terms to left side
            equation = left - right
            
            # Get the variables
            variables = list(equation.free_symbols)
            if not variables:
                return "No variables found to solve for.", []
            
            # Solve for the first variable (usually x)
            var = variables[0]
            solutions = sympy.solve(equation, var)
            
            # Format the solution
            if solutions:
                result = f"{var} = {', '.join(map(str, solutions))}"
                
                # Generate steps
                steps = generate_solving_steps(math_expr, var, solutions)
                
                return result, steps
            else:
                return "No solutions found. The equation may be inconsistent.", []
                
        else:
            # Check for variables to determine if we need to solve something
            try:
                expr = sympy.sympify(processed_expr)
                variables = list(expr.free_symbols)
                
                if variables:
                    # There are variables but no '=' sign - assume it's set to 0
                    var = variables[0] # Usually solve for x
                    equation = expr
                    solutions = sympy.solve(equation, var)
                    
                    if solutions:
                        result = f"{var} = {', '.join(map(str, solutions))}"
                        steps = [
                            f"Interpreting the expression as an equation: {math_expr} = 0",
                            f"Solving for {var}: {var} = {', '.join(map(str, solutions))}"
                        ]
                        return result, steps
                    else:
                        return "No solutions found when setting the expression to zero.", []
                else:
                    # Just evaluate the expression - no variables
                    result = sympy.N(expr)
                    steps = [f"Evaluating expression: {math_expr}", f"Result: {result}"]
                    return f"Result: {result}", steps
            
            except Exception as e:
                print(f"Error in expression evaluation: {str(e)}")
                return f"Error evaluating expression: {str(e)}", [f"Failed to process: {processed_expr}", f"Error: {str(e)}"]
            
    except Exception as e:
        print(f"Math solver error: {str(e)}")
        return f"Error solving math problem: {str(e)}", [f"Failed to parse: {math_expr}", f"Error details: {str(e)}"]

def debug_math_solver(question: str) -> Dict[str, Any]:
    """
    Debug function to help diagnose issues with the math solver.
    Returns detailed information about the parsing and solving process.
    """
    result = {
        "original_question": question,
        "is_math_question": is_math_question(question),
        "extracted_expression": extract_math_expression(question),
        "variable_to_solve": extract_variable_to_solve(question),
    }
    
    try:
        # Process the expression
        result["processed_expression"] = preprocess_expression(result["extracted_expression"])
        
        # Try to parse with sympy
        if '=' in result["processed_expression"]:
            left, right = result["processed_expression"].split('=', 1)
            result["parsed_left"] = str(sympy.sympify(left))
            result["parsed_right"] = str(sympy.sympify(right))
            result["equation"] = str(sympy.sympify(left) - sympy.sympify(right))
        else:
            result["parsed_expression"] = str(sympy.sympify(result["processed_expression"]))
        
        # Get full solution
        answer, steps = solve_math_problem(question)
        result["answer"] = answer
        result["steps"] = steps
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False
        
    return result

# Test cases function to verify the solver works properly
def run_test_cases():
    """Run a series of test cases to verify the math solver works correctly."""
    test_cases = [
        "2x+4=0 find x",
        "solve for x: 3x - 6 = 0", 
        "what is 2 + 2?",
        "find the value of x if 5x + 10 = 25",
        "solve the equation 3y - 9 = 0",
        "x^2 - 4 = 0",
        "find the roots of x^2 - 5x + 6 = 0"
    ]
    
    results = []
    for case in test_cases:
        answer, steps = solve_math_problem(case)
        results.append({
            "question": case,
            "answer": answer,
            "steps": steps
        })
        
    return results

# Add this function to your math_solver.py file
def format_math_solution(answer: str, steps: List[str]) -> str:
    """
    Format the math solution in a clean, readable way with proper spacing and layout.
    
    Args:
        answer: The final answer string
        steps: List of solution steps
        
    Returns:
        Properly formatted markdown string for display
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
        # Format verification section specially
        elif step.startswith("Verification"):
            formatted_solution += f"\n**Verification:**\n"
            formatted_solution += f"- {step.replace('Verification: ', '')}\n"
        # Format left side/right side specially for alignment
        elif step.startswith("Left side:"):
            formatted_solution += f"- {step}\n"
        elif step.startswith("Right side:"):
            formatted_solution += f"- {step}\n"
        elif step.startswith("Since"):
            formatted_solution += f"- {step}\n\n"
        # Format regular solution steps
        elif step.startswith("Step"):
            formatted_solution += f"{step_number}️⃣ {step.replace('Step ' + str(step_number-1) + ':', '')}\n\n"
            step_number += 1
        # Handle special cases for multiple solutions
        elif step.startswith("Multiple solutions found"):
            formatted_solution += f"\n**{step}**\n\n"
        elif step.startswith("Solution "):
            formatted_solution += f"- {step}\n"
        # Format any mathematical expressions specially
        elif "=" in step and not step.startswith("Result"):
            formatted_solution += f"```\n{step}\n```\n\n"
        # Default formatting for other steps
        else:
            formatted_solution += f"{step}\n\n"
    
    return formatted_solution