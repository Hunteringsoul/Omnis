import os
from openai import OpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import tiktoken
from functools import lru_cache
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
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
        # Calculate cost (GPT-3.5-turbo: $0.002 per 1K tokens)
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

# Initialize cost tracker
cost_tracker = CostTracker()

# Cached Wikipedia wrapper
@lru_cache(maxsize=1)
def get_wikipedia():
    return WikipediaAPIWrapper()

def call_openrouter(prompt, system_prompt="You are a helpful assistant that can answer questions and perform tasks."):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenRouter: {str(e)}")
        return f"Error: {str(e)}"

def process_query(query):
    try:
        # Get Wikipedia data if needed
        wikipedia = get_wikipedia()
        wiki_data = wikipedia.run(query)
        
        # Create a comprehensive prompt
        prompt = f"""
You are an AI assistant that can answer questions and perform tasks.
If the question requires factual information, use the Wikipedia data provided.
If the question is about a task or process, provide step-by-step instructions.

Question: {query}

Wikipedia Information (if relevant):
{wiki_data}

Please provide a comprehensive answer:
"""
        
        # Call OpenRouter with the prompt
        system_prompt = "You are a helpful assistant that can answer questions and perform tasks."
        return call_openrouter(prompt, system_prompt)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"I encountered an error while processing your query: {str(e)}"

def get_executor():
    """
    Returns a function that can be used to process queries.
    This maintains compatibility with the existing code structure.
    """
    def executor(query_dict):
        query = query_dict.get("input", "")
        response = process_query(query)
        return {"output": response}
    
    return executor

# Test the agent executor
if __name__ == "__main__":
    print("Initializing Executor Agent...")
    executor = get_executor()
    
    print("\nExecutor Agent CLI")
    print("=================")
    print("Type 'exit' to quit")
    print("Ask any question!")
    print("===============================================\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nExiting...")
                break
            
            if not user_input:
                continue
            
            print("\nProcessing your query...")
            
            # Track input tokens
            input_tokens = count_tokens(user_input)
            cost_tracker.track_usage(input_tokens)
            
            # Process the query
            response = executor({"input": user_input})
            
            # Track output tokens
            output_tokens = count_tokens(response["output"])
            cost_tracker.track_usage(output_tokens)
            
            # Print response with usage info
            print(f"\nResponse: {response['output']}")
            print(f"\nUsage: {input_tokens + output_tokens} tokens (${(input_tokens + output_tokens)/1000*0.002:.4f})")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")
