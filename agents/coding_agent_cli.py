import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure OpenRouter
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

# Initialize token counter
def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# Cost tracking
class CostTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0
        self.daily_usage = {}
        self.load_usage()

    def load_usage(self):
        if os.path.exists('coding_usage.json'):
            with open('coding_usage.json', 'r') as f:
                data = json.load(f)
                self.total_tokens = data.get('total_tokens', 0)
                self.total_cost = data.get('total_cost', 0)
                self.daily_usage = data.get('daily_usage', {})

    def save_usage(self):
        with open('coding_usage.json', 'w') as f:
            json.dump({
                'total_tokens': self.total_tokens,
                'total_cost': self.total_cost,
                'daily_usage': self.daily_usage
            }, f)

    def track_usage(self, tokens: int):
        today = datetime.now().strftime('%Y-%m-%d')
        self.total_tokens += tokens
        self.daily_usage[today] = self.daily_usage.get(today, 0) + tokens
        # Calculate cost (OpenRouter GPT-3.5-turbo: $0.002 per 1K tokens)
        self.total_cost = (self.total_tokens / 1000) * 0.002
        self.save_usage()

    def get_usage_summary(self):
        return f"""
Usage Summary:
-------------
Total Tokens: {self.total_tokens:,}
Total Cost: ${self.total_cost:.4f}
Today's Usage: {self.daily_usage.get(datetime.now().strftime('%Y-%m-%d'), 0):,} tokens
"""

def generate_code(prompt, language="python"):
    system_prompt = f"""You are an expert programmer specializing in {language}. 
    Generate clean, efficient, and well-documented code based on the user's request.
    Include necessary imports and explain the code with comments.
    Focus on best practices and maintainable code."""
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def explain_code(code):
    system_prompt = """You are a code explanation expert. 
    Explain the provided code in a clear and concise way.
    Focus on:
    1. The overall purpose of the code
    2. Key components and their functions
    3. Important algorithms or patterns used
    4. Potential improvements or optimizations"""
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": code}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def debug_code(code, error_message=None):
    system_prompt = """You are a debugging expert. 
    Analyze the code and identify potential issues or bugs.
    If an error message is provided, focus on that specific error.
    Provide:
    1. The identified issues
    2. The root cause
    3. Suggested fixes
    4. Prevention tips for future"""
    
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
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("Initializing Coding Agent...")
    cost_tracker = CostTracker()
    
    print("\nCoding Agent CLI")
    print("===============")
    print("Commands:")
    print("  generate <language> <prompt> - Generate code in specified language")
    print("  explain <code> - Get explanation of the code")
    print("  debug <code> [error_message] - Debug code with optional error message")
    print("  usage - Show usage statistics")
    print("  exit - Exit the program")
    print("===============================================\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nFinal Usage Summary:")
                print(cost_tracker.get_usage_summary())
                break
                
            if user_input.lower() == 'usage':
                print(cost_tracker.get_usage_summary())
                continue

            if not user_input:
                continue

            # Track input tokens
            input_tokens = count_tokens(user_input)
            cost_tracker.track_usage(input_tokens)

            # Process the command
            if user_input.lower().startswith('generate '):
                parts = user_input[9:].split(' ', 1)
                if len(parts) == 2:
                    language, prompt = parts
                    print(f"\nGenerating {language} code...")
                    response = generate_code(prompt, language)
                    print(f"\nGenerated Code:\n{response}")
                else:
                    print("\nPlease use format: generate <language> <prompt>")
                    continue
                    
            elif user_input.lower().startswith('explain '):
                code = user_input[8:].strip()
                if code:
                    print("\nAnalyzing code...")
                    response = explain_code(code)
                    print(f"\nExplanation:\n{response}")
                else:
                    print("\nPlease provide code to explain.")
                    continue
                    
            elif user_input.lower().startswith('debug '):
                parts = user_input[6:].split(' ', 1)
                if len(parts) >= 1:
                    code = parts[0]
                    error_message = parts[1] if len(parts) > 1 else None
                    print("\nDebugging code...")
                    response = debug_code(code, error_message)
                    print(f"\nDebug Analysis:\n{response}")
                else:
                    print("\nPlease provide code to debug.")
                    continue
                    
            else:
                print("\nUnknown command. Available commands:")
                print("  generate <language> <prompt> - Generate code in specified language")
                print("  explain <code> - Get explanation of the code")
                print("  debug <code> [error_message] - Debug code with optional error message")
                print("  usage - Show usage statistics")
                print("  exit - Exit the program")
                continue

            # Track output tokens
            output_tokens = count_tokens(str(response))
            cost_tracker.track_usage(output_tokens)

            # Print usage info
            print(f"\nUsage: {input_tokens + output_tokens:,} tokens (${(input_tokens + output_tokens)/1000*0.002:.4f})")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            print(cost_tracker.get_usage_summary())
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main() 