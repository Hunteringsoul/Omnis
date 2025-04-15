from openai import OpenAI
from langchain_community.utilities import WikipediaAPIWrapper
import tiktoken
from functools import lru_cache
import json
from datetime import datetime
import os

class MathAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.cost_tracker = CostTracker()
        
    def solve(self, problem: str) -> str:
        """Solve a math problem using the OpenAI API."""
        # Track input tokens
        input_tokens = count_tokens(problem)
        self.cost_tracker.track_usage(input_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a math expert. Solve the given math problem step by step. Show your work clearly and provide the final answer."},
                    {"role": "user", "content": problem}
                ],
                temperature=0,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Track output tokens
            output_tokens = count_tokens(answer)
            self.cost_tracker.track_usage(output_tokens)
            
            # Add usage info to response
            usage_info = f"\n\n---\n*Usage: {input_tokens + output_tokens} tokens (${(input_tokens + output_tokens)/1000*0.002:.4f})*"
            return answer + usage_info
            
        except Exception as e:
            return f"Error solving math problem: {str(e)}"

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

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