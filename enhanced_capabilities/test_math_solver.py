import sys
from math_solver import extract_math_expression, solve_math_problem

def test_extraction():
    test_cases = [
        "2x+4=0 find x",
        "2x+4=0find x",
        "solve for x: 3x - 6 = 0",
        "find the value of x if 5x + 10 = 25"
    ]
    
    for test in test_cases:
        extracted = extract_math_expression(test)
        print(f"Original: '{test}'")
        print(f"Extracted: '{extracted}'")
        print("-" * 40)

def test_solver():
    test_cases = [
        "2x+4=0 find x",
        "x^2 - 4 = 0",
        "solve for x: 3x - 6 = 0"
    ]
    
    for test in test_cases:
        answer, steps = solve_math_problem(test)
        print(f"Question: '{test}'")
        print(f"Answer: {answer}")
        print("Steps:")
        for step in steps:
            print(f"  - {step}")
        print("-" * 40)

if __name__ == "__main__":
    print("Testing extraction:")
    test_extraction()
    print("\nTesting solver:")
    test_solver()