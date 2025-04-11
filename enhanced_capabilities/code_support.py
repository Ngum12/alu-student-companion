import re
import ast
from typing import Dict, Any, List, Tuple, Optional
import difflib
from duckduckgo_search import DDGS

def is_code_question(question: str) -> bool:
    """Determine if a question is related to code."""
    code_keywords = [
        "code", "function", "algorithm", "syntax", "error", "debug", 
        "compile", "runtime", "python", "javascript", "java", "c++", "html", 
        "css", "sql", "program", "variable", "class", "object", "method",
        "array", "list", "dictionary", "loop", "if statement", "condition",
        "exception", "try", "catch", "import", "module", "library",
        "fetch", "api", "request", "response", "html", "tag", "element"
    ]
    
    # Check if any code keywords are in the question
    has_keywords = any(keyword.lower() in question.lower() for keyword in code_keywords)
    
    # Check for code block markers
    has_code_blocks = '```' in question or '`' in question
    
    # Check for common code patterns
    code_patterns = [
        r'def\s+\w+\s*\(', # Python function definition
        r'function\s+\w+\s*\(', # JavaScript function definition
        r'class\s+\w+', # Class definition
        r'for\s+\w+\s+in\s+', # Python for loop
        r'if\s+.+:', # Python if statement
        r'import\s+\w+', # Import statement
        r'from\s+\w+\s+import', # Python from import
        r'<\w+>.*</\w+>', # HTML tags
        r'<\w+[^>]*>', # HTML opening tags
        r'SELECT\s+.+\s+FROM\s+', # SQL query
        r'var\s+\w+\s*=', # JavaScript var declaration
        r'let\s+\w+\s*=', # JavaScript let declaration
        r'const\s+\w+\s*=' # JavaScript const declaration
    ]
    
    has_code_patterns = any(re.search(pattern, question, re.IGNORECASE) for pattern in code_patterns)
    
    return has_keywords or has_code_blocks or has_code_patterns

def extract_code(text: str) -> List[Tuple[str, str]]:
    """Extract code blocks from text with their language."""
    # Extract markdown code blocks with language specification
    code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', text, re.DOTALL)
    
    # If no code blocks with language specification were found,
    # look for code blocks without language specification
    if not code_blocks:
        unlabeled_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        code_blocks = [('', block.strip()) for block in unlabeled_blocks]
    
    # If still no code blocks, look for inline code
    if not code_blocks:
        inline_code = re.findall(r'`(.*?)`', text)
        code_blocks = [('', code.strip()) for code in inline_code]
        
    # If still nothing, try to find code-like patterns
    if not code_blocks:
        # Look for indented blocks that might be code
        lines = text.split('\n')
        current_block = []
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                current_block.append(line.lstrip())
            elif current_block:
                # We found a non-indented line after some indented lines
                code_blocks.append(('', '\n'.join(current_block)))
                current_block = []
                
        if current_block:  # Don't forget the last block
            code_blocks.append(('', '\n'.join(current_block)))

        # Look for HTML-like content
        if '<' in text and '>' in text:
            html_tags = re.findall(r'(<[^>]+>.*?</[^>]+>)', text, re.DOTALL)
            if html_tags:
                code_blocks.append(('html', html_tags[0]))
            
    return code_blocks

def guess_language(code: str) -> str:
    """Guess the programming language of a code snippet."""
    # Simple heuristics to identify common languages
    # Python patterns
    if re.search(r'def\s+\w+\s*\(.*\):', code) or \
       re.search(r'import\s+\w+', code) or \
       re.search(r'from\s+\w+\s+import', code) or \
       re.search(r'if\s+.*:', code) or \
       re.search(r'for\s+.*\s+in\s+.*:', code) or \
       re.search(r'class\s+\w+.*:', code):
        return 'python'
    
    # JavaScript patterns
    elif re.search(r'function\s+\w+\s*\(.*\)', code) or \
         re.search(r'const\s+\w+\s*=', code) or \
         re.search(r'let\s+\w+\s*=', code) or \
         re.search(r'var\s+\w+\s*=', code) or \
         re.search(r'export\s+class', code) or \
         re.search(r'=>', code):
        return 'javascript'
    
    # Java patterns
    elif re.search(r'public\s+static\s+void\s+main', code) or \
         re.search(r'public\s+class\s+\w+', code):
        return 'java'
    
    # C/C++ patterns
    elif re.search(r'#include\s*<\w+\.h>', code) or \
         re.search(r'int\s+main\s*\(\s*void\s*\)', code) or \
         re.search(r'int\s+main\s*\(\s*int\s+argc,\s*char\s*\*\s*argv\[\]\s*\)', code):
        return 'c/c++'
    
    # HTML patterns
    elif re.search(r'<html>', code, re.IGNORECASE) or \
         re.search(r'<body>', code, re.IGNORECASE) or \
         re.search(r'<div', code, re.IGNORECASE) or \
         re.search(r'<p>', code, re.IGNORECASE):
        return 'html'
    
    # SQL patterns
    elif re.search(r'SELECT\s+.*\s+FROM\s+', code, re.IGNORECASE) or \
         re.search(r'INSERT\s+INTO', code, re.IGNORECASE) or \
         re.search(r'CREATE\s+TABLE', code, re.IGNORECASE):
        return 'sql'
    
    # Default to Python for code blocks with indentation and common Python syntax
    # This helps catch Python code without explicit markers
    elif ':' in code and ('def ' in code or 'if ' in code or 'for ' in code or 'while ' in code):
        return 'python'
        
    return 'unknown'

def analyze_python_code(code: str) -> Dict[str, Any]:
    """Analyze Python code to extract structure information."""
    try:
        tree = ast.parse(code)
        
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module} (from)")
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "valid_syntax": True
        }
    except SyntaxError as e:
        return {
            "valid_syntax": False,
            "error": str(e),
            "error_line": e.lineno,
            "error_offset": e.offset
        }
    except Exception as e:
        return {
            "valid_syntax": False,
            "error": str(e)
        }

def fix_common_python_errors(code: str) -> Tuple[str, List[str]]:
    """Fix common Python syntax errors."""
    original_code = code
    fixes = []
    
    # Fix 1: Missing colons after if/for/while/def statements
    def add_missing_colons(code):
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Check for control structures without colons
            if re.search(r'^\s*(if|elif|else|for|while|def|class|with|try|except|finally)\s+.*[^\s:]$', line):
                lines[i] = line + ':'
                return '\n'.join(lines), f"Added missing colon at line {i+1}"
        return code, None
    
    # Fix 2: Indentation errors
    def fix_indentation(code):
        lines = code.split('\n')
        fixed_lines = []
        
        # Simple heuristic: ensure consistent indentation of 4 spaces
        current_indent = 0
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue
                
            # Lines that typically increase indentation
            if re.search(r':\s*$', line):
                fixed_lines.append(' ' * current_indent + stripped)
                current_indent += 4
            # Lines that typically decrease indentation
            elif stripped.startswith(('else:', 'elif ', 'except:', 'finally:', 'except ')):
                current_indent = max(0, current_indent - 4)
                fixed_lines.append(' ' * current_indent + stripped)
                current_indent += 4  # Indent again after these statements
            else:
                fixed_lines.append(' ' * current_indent + stripped)
                
        fixed_code = '\n'.join(fixed_lines)
        if fixed_code != code:
            return fixed_code, "Fixed inconsistent indentation"
        return code, None
    
    # Fix 3: Unmatched parentheses/brackets/braces
    def fix_unmatched_brackets(code):
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets.keys():
                stack.append(char)
            elif char in brackets.values():
                if not stack or char != brackets[stack.pop()]:
                    # Unbalanced closing bracket found
                    pass
        
        if stack:
            # Add missing closing brackets
            fixed_code = code
            for bracket in reversed(stack):
                fixed_code += brackets[bracket]
            return fixed_code, f"Added missing {len(stack)} closing bracket(s)"
        
        return code, None
    
    # Fix 4: Missing quotes in strings
    def fix_missing_quotes(code):
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Look for string assignments without proper quotes
            match = re.search(r'=\s*(\w+)$', line)
            if match and match.group(1) not in ['True', 'False', 'None']:
                potential_string = match.group(1)
                lines[i] = line.replace(potential_string, f'"{potential_string}"')
                return '\n'.join(lines), f"Added missing quotes around '{potential_string}' at line {i+1}"
        return code, None
    
    # Apply fixes
    for fix_func in [add_missing_colons, fix_indentation, fix_unmatched_brackets, fix_missing_quotes]:
        code, message = fix_func(code)
        if message:
            fixes.append(message)
    
    # If we couldn't fix it with our simple heuristics, return the original
    if not fixes:
        return original_code, []
    
    return code, fixes

def explain_python_code(code: str) -> str:
    """Generate an explanation for Python code."""
    try:
        explanation = []
        tree = ast.parse(code)
        
        # Get imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module}")
                
        if imports:
            explanation.append("**Imports:**")
            explanation.append("This code imports the following modules: " + ", ".join(imports) + ".")
        
        # Analyze functions
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if functions:
            explanation.append("\n**Functions:**")
            for func in functions:
                args = [arg.arg for arg in func.args.args]
                explanation.append(f"- `{func.name}({', '.join(args)})`: ")
                
                # Try to determine what the function does
                func_operations = []
                for node in ast.walk(func):
                    if isinstance(node, ast.Return):
                        func_operations.append("returns a value")
                        break
                
                func_body = ast.unparse(func).split(':', 1)[1].strip()
                if func_operations:
                    explanation.append(f"  This function {' and '.join(func_operations)}.")
                else:
                    explanation.append(f"  This function takes {len(args)} parameter(s).")
        
        # Analyze classes
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        if classes:
            explanation.append("\n**Classes:**")
            for cls in classes:
                base_names = [ast.unparse(base) for base in cls.bases]
                if base_names:
                    explanation.append(f"- `{cls.name}`: A class that inherits from {', '.join(base_names)}.")
                else:
                    explanation.append(f"- `{cls.name}`: A class definition.")
                
                # Look for methods
                methods = [node.name for node in ast.walk(cls) if isinstance(node, ast.FunctionDef)]
                if methods:
                    explanation.append(f"  It has the following methods: {', '.join(methods)}.")
        
        # Look for main execution
        has_main = "if __name__ == '__main__':" in code
        if has_main:
            explanation.append("\n**Program Structure:**")
            explanation.append("This code includes a main execution block that runs when the script is executed directly.")
        
        if not explanation:
            explanation.append("This is a simple Python script.")
            
        return "\n".join(explanation)
        
    except Exception as e:
        # Provide a more general explanation if parsing failed
        lines = code.split('\n')
        explanation = [f"This code consists of {len(lines)} lines of Python."]
        
        # Look for patterns we can describe
        if "def " in code:
            explanation.append("It appears to define one or more functions.")
        if "class " in code:
            explanation.append("It contains class definitions.")
        if "import " in code:
            explanation.append("It imports external modules.")
        if "for " in code:
            explanation.append("It uses loops to iterate through data.")
        if "if " in code:
            explanation.append("It contains conditional logic.")
            
        explanation.append("(Note: I couldn't fully parse the code structure.)")
        return "\n".join(explanation)

def search_code_solutions(query: str) -> Dict[str, Any]:
    """Search for code solutions online."""
    # Add "code" and "example" to the query to get more relevant results
    search_query = f"{query} code example"
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=5))
        
        if not results:
            return {
                "found": False,
                "message": "I couldn't find any code examples related to your query online."
            }
        
        # Extract relevant snippets that might contain code
        snippets = [result['body'] for result in results]
        links = [result['href'] for result in results]
        
        # Extract titles for better context
        titles = [result.get('title', 'Resource') for result in results]
        
        return {
            "found": True,
            "snippets": snippets,
            "links": links,
            "titles": titles
        }
    
    except Exception as e:
        return {
            "found": False,
            "message": f"Error searching for code solutions: {str(e)}"
        }

# Fix the handle_code_question function to better handle generic code requests
def handle_code_question(question: str) -> Dict[str, Any]:
    """Handle code-related questions."""
    # Extract code blocks from the question
    code_blocks = extract_code(question)
    question_lower = question.lower()
    
    # Handle specific code generation requests
    if "html" in question_lower and any(word in question_lower for word in ["code", "basic", "template", "boilerplate"]):
        language, code = generate_simple_code("html template")
        return {
            "answer": f"Here's a basic HTML template:",
            "language": language,
            "code": code,
            "type": "generate"
        }
        
    elif "python" in question_lower and any(word in question_lower for word in ["code", "basic", "example"]):
        language, code = generate_simple_code("python example")
        return {
            "answer": f"Here's a simple Python example:",
            "language": language,
            "code": code,
            "type": "generate"
        }
    
    # Rest of your existing function...
    # If no code blocks found but it's a code question, look for keywords
    if not code_blocks and is_code_question(question):
        question_lower = question.lower()
        
        # Check if this is a request to generate code
        if any(keyword in question_lower for keyword in ["generate", "create", "write", "make"]):
            language, code = generate_simple_code(question)
            return {
                "answer": f"Here's a {language} code example that might help:",
                "language": language,
                "code": code,
                "type": "generate"
            }
        
        # Special case for JavaScript API fetch
        elif "fetch" in question_lower and ("api" in question_lower or "javascript" in question_lower):
            language, code = generate_simple_code("fetch api javascript")
            return {
                "answer": f"Here's how to fetch data from an API in JavaScript:",
                "language": "javascript",
                "code": code,
                "type": "generate"
            }
        
        # Special case for HTML issues
        elif "html" in question_lower and any(word in question_lower for word in ["wrong", "fix", "error", "issue"]):
            return {
                "answer": "Common HTML errors include:\n\n1. Mismatched tags (opening and closing tags don't match)\n2. Missing closing tags\n3. Incorrect nesting of elements\n4. Invalid attributes\n5. Duplicate IDs\n\nTo fix HTML issues, ensure all tags are properly closed and nested correctly.",
                "type": "explain"
            }
            
        # Check if this is a request to explain a programming concept
        elif any(keyword in question_lower for keyword in ["explain", "how does", "what is", "how to"]):
            # Search online for code examples related to the concept
            search_results = search_code_solutions(question)
            
            if search_results["found"]:
                # Format a response with links to resources
                answer = f"Here's what I found about {question}:\n\n"
                
                for i, (title, snippet, link) in enumerate(zip(
                    search_results["titles"][:3], 
                    search_results["snippets"][:3], 
                    search_results["links"][:3]
                )):
                    answer += f"**{title}**\n{snippet[:150]}...\n[View resource]({link})\n\n"
                
                return {
                    "answer": answer,
                    "type": "explain",
                    "resources": search_results["links"][:3]
                }
            else:
                return {
                    "answer": "I don't have enough information to explain this code concept. Please provide more details or a specific code example.",
                    "type": "explain"
                }
    
    # Handle code that was provided in the question
    if code_blocks:
        language, code = code_blocks[0]
        
        # If language wasn't specified, try to guess it
        if not language:
            language = guess_language(code)
        
        # For Python code, we can do more analysis
        if language.lower() == 'python' or guess_language(code) == 'python':
            # Check if it has syntax errors
            analysis = analyze_python_code(code)
            
            if not analysis.get("valid_syntax", True):
                # Try to fix the code
                fixed_code, fixes = fix_common_python_errors(code)
                
                if fixes:
                    explanation = explain_python_code(fixed_code)
                    
                    diff = list(difflib.unified_diff(
                        code.splitlines(keepends=True),
                        fixed_code.splitlines(keepends=True),
                        fromfile='original',
                        tofile='fixed'
                    ))
                    
                    return {
                        "answer": f"I found and fixed these issues in your code:\n- " + "\n- ".join(fixes),
                        "explanation": explanation,
                        "language": "python",
                        "original_code": code,
                        "fixed_code": fixed_code,
                        "type": "fix"
                    }
                else:
                    return {
                        "answer": f"There appears to be a syntax error in your code: {analysis.get('error', 'Unknown error')}\n\nUnfortunately, I couldn't automatically fix it.",
                        "language": "python",
                        "code": code,
                        "type": "error"
                    }
            else:
                # Explain the valid code
                explanation = explain_python_code(code)
                return {
                    "answer": "Here's an explanation of your Python code:",
                    "explanation": explanation,
                    "language": "python",
                    "code": code,
                    "type": "explain"
                }
        # For HTML code, provide error analysis
        elif language.lower() == 'html' or guess_language(code) == 'html':
            html_analysis = analyze_html_errors(code)
            
            if html_analysis["has_errors"]:
                return {
                    "answer": "I found these issues in your HTML code:\n- " + "\n- ".join(html_analysis["errors"]),
                    "language": "html",
                    "code": code,
                    "type": "fix"
                }
            else:
                return {
                    "answer": "This appears to be valid HTML code.",
                    "language": "html",
                    "code": code,
                    "type": "identify"
                }
        else:
            # For other languages, provide a more general response
            return {
                "answer": f"This appears to be {language} code. Unfortunately, I can only fully analyze Python code right now.",
                "language": language,
                "code": code,
                "type": "identify"
            }
    
    # Default fallback
    return {
        "answer": "I can help with code questions, but I need you to provide a specific code snippet or a clear description of what you want to generate.",
        "type": "fallback"
    }

def generate_simple_code(request: str) -> Tuple[str, str]:
    """Generate simple code snippets based on the request."""
    request_lower = request.lower()
    
    # JavaScript API fetch function
    if "fetch api javascript" in request_lower or ("api" in request_lower and "fetch" in request_lower):
        return "javascript", """// Function to fetch data from an API
async function fetchData(url) {
    try {
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
        throw error;
    }
}

// Example usage
const apiUrl = 'https://api.example.com/data';

// Using the function with async/await
async function displayData() {
    try {
        const result = await fetchData(apiUrl);
        console.log('Data received:', result);
        
        // Do something with the data
        document.getElementById('result').textContent = JSON.stringify(result, null, 2);
    } catch (error) {
        console.error('Failed to get data:', error);
    }
}

// Call the function
displayData();
"""
    
    # Python function to check if a number is prime
    elif "prime" in request_lower and ("check" in request_lower or "number" in request_lower):
        return "python", """def is_prime(n):
    \"\"\"Check if a number is prime.\"\"\"
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Example usage
number = 17
if is_prime(number):
    print(f"{number} is a prime number")
else:
    print(f"{number} is not a prime number")
"""
    
    # Python function to calculate factorial
    elif "factorial" in request_lower:
        return "python", """def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Example usage
number = 5
print(f"The factorial of {number} is {factorial(number)}")
"""
    
    # Python function to find the Fibonacci sequence
    elif "fibonacci" in request_lower:
        return "python", """def fibonacci(n):
    \"\"\"Generate the Fibonacci sequence up to the nth term.\"\"\"
    sequence = []
    a, b = 0, 1
    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence

# Example usage
n_terms = 10
print(f"Fibonacci sequence with {n_terms} terms: {fibonacci(n_terms)}")
"""
    
    # HTML boilerplate
    elif "html" in request_lower and ("boilerplate" in request_lower or "template" in request_lower):
        return "html", """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Website</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        header {
            background-color: #f4f4f4;
            padding: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Welcome to My Website</h1>
        </header>
        <main>
            <p>This is a simple HTML template to get you started.</p>
        </main>
        <footer>
            <p>&copy; 2025 My Website. All rights reserved.</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # SQL query to create a table and insert data
    elif "sql" in request_lower and ("table" in request_lower or "database" in request_lower):
        return "sql", """-- Create a users table
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    is_active BOOLEAN DEFAULT TRUE
);

-- Insert sample data
INSERT INTO users (username, email, password_hash) VALUES
('john_doe', 'john@example.com', 'hashed_password_1'),
('jane_smith', 'jane@example.com', 'hashed_password_2'),
('bob_johnson', 'bob@example.com', 'hashed_password_3');

-- Query to select all active users
SELECT user_id, username, email, created_at 
FROM users
WHERE is_active = TRUE
ORDER BY created_at DESC;
"""
    
    # Default response for other code requests
    else:
        return "python", """# Here's a simple Python example
def greet(name):
    \"\"\"Return a greeting message.\"\"\"
    return f"Hello, {name}! Welcome to programming."

# Example usage
user_name = "World"
message = greet(user_name)
print(message)
"""

def analyze_html_errors(html_code: str) -> Dict[str, Any]:
    """Detect common HTML errors."""
    errors = []
    
    # Check for mismatched tags
    opening_tags = re.findall(r'<(\w+)[^>]*>', html_code)
    closing_tags = re.findall(r'</(\w+)>', html_code)
    
    # Simple check for matching opening and closing tags
    if len(opening_tags) != len(closing_tags):
        errors.append("Mismatched number of opening and closing tags")
    
    # Check for specific mismatches
    for i, tag in enumerate(closing_tags):
        if i < len(opening_tags) and tag != opening_tags[-(i+1)]:
            errors.append(f"Tag mismatch: Opening <{opening_tags[-(i+1)]}>  doesn't match closing </{tag}>")
    
    # Check for unclosed tags
    for tag in opening_tags:
        if tag not in closing_tags and tag not in ['img', 'br', 'hr', 'input', 'meta', 'link']:
            errors.append(f"Unclosed tag: <{tag}>")
    
    # Check for invalid attributes
    if re.search(r'<\w+\s+[^>]*=\s*[^\'"][^\s>]*[^\'"][^>]*>', html_code):
        errors.append("Possible invalid attribute format (missing quotes)")
    
    return {
        "has_errors": len(errors) > 0,
        "errors": errors
    }