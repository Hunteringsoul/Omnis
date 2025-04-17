import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.utilities import WikipediaAPIWrapper
import tiktoken
from functools import lru_cache
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
        if os.path.exists('usage.json'):
            with open('usage.json', 'r') as f:
                data = json.load(f)
                self.total_tokens = data.get('total_tokens', 0)
                self.total_cost = data.get('total_cost', 0)
                self.daily_usage = data.get('daily_usage', {})

    def save_usage(self):
        with open('usage.json', 'w') as f:
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

def call_openrouter(prompt, system_prompt="You are a helpful math assistant."):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def solve_math_problem(problem):
    system_prompt = """You are a math expert. Solve the given math problem step by step.
    Show your work clearly and provide the final answer."""
    return call_openrouter(problem, system_prompt)

def solve_word_problem(problem):
    system_prompt = """You are a reasoning agent tasked with solving logic-based questions.
    Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved
    and give the final answer. Provide the response in bullet points."""
    return call_openrouter(problem, system_prompt)

def get_wikipedia_info(query):
    try:
        wiki = WikipediaAPIWrapper()
        return wiki.run(query)
    except Exception as e:
        return f"Error accessing Wikipedia: {str(e)}"

def main():
    print("Initializing Math Agent...")
    cost_tracker = CostTracker()
    
    print("\nMath Agent CLI")
    print("=============")
    print("Type 'exit' to quit, 'usage' to see usage statistics")
    print("Ask any math, logic, or general knowledge question!")
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

            # Determine the type of question and process accordingly
            if any(op in user_input for op in ['+', '-', '*', '/', '=', '^', 'sqrt']):
                response = solve_math_problem(user_input)
            elif 'what is' in user_input.lower() or 'who is' in user_input.lower() or 'when' in user_input.lower():
                response = get_wikipedia_info(user_input)
            else:
                response = solve_word_problem(user_input)

            # Track output tokens
            output_tokens = count_tokens(response)
            cost_tracker.track_usage(output_tokens)

            # Print response with usage info
            print("\nAgent:", response)
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