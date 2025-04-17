import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import json
from datetime import datetime
from .coding_models import CodeResponse, CodeExplanation, CodeDebug

# Load environment variables
load_dotenv()

# Configure OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def generate_structured_code(prompt: str, language: str = "python") -> CodeResponse:
    """Generate code with structured output using Pydantic models"""
    system_prompt = f"""You are an expert programmer specializing in {language}. 
    Generate clean, efficient, and well-documented code based on the user's request.
    Your response must be a valid JSON object with the following structure:
    {{
        "code": "the actual code",
        "language": "{language}",
        "imports": ["required imports"],
        "explanation": "brief explanation",
        "complexity": "time and space complexity"
    }}
    Focus on best practices and maintainable code."""
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Parse the response into a CodeResponse object
        code_response = CodeResponse.model_validate_json(response.choices[0].message.content)
        return code_response
    except Exception as e:
        return CodeResponse(
            code=f"# Error: {str(e)}",
            language=language,
            imports=[],
            explanation="An error occurred while generating the code.",
            complexity=None
        )

def explain_structured_code(code: str) -> CodeExplanation:
    """Explain code with structured output using Pydantic models"""
    system_prompt = """You are a code explanation expert. 
    Explain the provided code in a clear and concise way.
    Your response must be a valid JSON object with the following structure:
    {
        "purpose": "overall purpose",
        "components": ["key components"],
        "algorithms": ["algorithms used"],
        "improvements": ["potential improvements"]
    }"""
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": code}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Parse the response into a CodeExplanation object
        explanation = CodeExplanation.model_validate_json(response.choices[0].message.content)
        return explanation
    except Exception as e:
        return CodeExplanation(
            purpose="Error occurred while analyzing code",
            components=[f"Error: {str(e)}"],
            algorithms=None,
            improvements=None
        )

def debug_structured_code(code: str, error_message: str = None) -> CodeDebug:
    """Debug code with structured output using Pydantic models"""
    system_prompt = """You are a debugging expert. 
    Analyze the code and identify potential issues or bugs.
    Your response must be a valid JSON object with the following structure:
    {
        "issues": ["identified issues"],
        "root_cause": "root cause",
        "fixes": ["suggested fixes"],
        "prevention": ["prevention tips"]
    }"""
    
    try:
        user_content = f"Code:\n{code}\n"
        if error_message:
            user_content += f"\nError message:\n{error_message}"
            
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Parse the response into a CodeDebug object
        debug_info = CodeDebug.model_validate_json(response.choices[0].message.content)
        return debug_info
    except Exception as e:
        return CodeDebug(
            issues=[f"Error: {str(e)}"],
            root_cause="Error occurred while debugging",
            fixes=["Please try again"],
            prevention=["Ensure proper error handling"]
        )

def format_code_response(code_response: CodeResponse) -> str:
    """Format the code response for display"""
    output = []
    
    # Add imports
    if code_response.imports:
        output.append("# Required imports:")
        for imp in code_response.imports:
            output.append(imp)
        output.append("")
    
    # Add code
    output.append(code_response.code)
    output.append("")
    
    # Add explanation if available
    if code_response.explanation:
        output.append("# Explanation:")
        output.append(code_response.explanation)
        output.append("")
    
    # Add complexity if available
    if code_response.complexity:
        output.append("# Complexity:")
        output.append(code_response.complexity)
    
    return "\n".join(output)

def format_explanation(explanation: CodeExplanation) -> str:
    """Format the code explanation for display"""
    output = []
    
    output.append("Purpose:")
    output.append(explanation.purpose)
    output.append("")
    
    output.append("Key Components:")
    for component in explanation.components:
        output.append(f"- {component}")
    output.append("")
    
    if explanation.algorithms:
        output.append("Algorithms/Patterns Used:")
        for algo in explanation.algorithms:
            output.append(f"- {algo}")
        output.append("")
    
    if explanation.improvements:
        output.append("Potential Improvements:")
        for imp in explanation.improvements:
            output.append(f"- {imp}")
    
    return "\n".join(output)

def format_debug_info(debug_info: CodeDebug) -> str:
    """Format the debug information for display"""
    output = []
    
    output.append("Identified Issues:")
    for issue in debug_info.issues:
        output.append(f"- {issue}")
    output.append("")
    
    output.append("Root Cause:")
    output.append(debug_info.root_cause)
    output.append("")
    
    output.append("Suggested Fixes:")
    for fix in debug_info.fixes:
        output.append(f"- {fix}")
    output.append("")
    
    output.append("Prevention Tips:")
    for tip in debug_info.prevention:
        output.append(f"- {tip}")
    
    return "\n".join(output) 