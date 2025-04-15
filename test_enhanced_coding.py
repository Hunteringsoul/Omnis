from agents.enhanced_coding_agent import (
    generate_structured_code,
    explain_structured_code,
    debug_structured_code,
    format_code_response,
    format_explanation,
    format_debug_info
)

def test_code_generation():
    print("Testing code generation...")
    print("="*50)
    
    # Test code generation
    prompt = "Create a function to find the factorial of a number"
    response = generate_structured_code(prompt, "python")
    print(format_code_response(response))
    print("="*50)

def test_code_explanation():
    print("\nTesting code explanation...")
    print("="*50)
    
    # Test code explanation
    code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
    """
    explanation = explain_structured_code(code)
    print(format_explanation(explanation))
    print("="*50)

def test_code_debugging():
    print("\nTesting code debugging...")
    print("="*50)
    
    # Test code debugging
    code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # This will raise ZeroDivisionError if numbers is empty
    """
    error_message = "ZeroDivisionError: division by zero"
    debug_info = debug_structured_code(code, error_message)
    print(format_debug_info(debug_info))
    print("="*50)

if __name__ == "__main__":
    test_code_generation()
    test_code_explanation()
    test_code_debugging() 