import os
import chainlit as cl
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

def call_openrouter(prompt, system_prompt="You are a helpful research assistant."):
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

def research_topic(topic):
    try:
        # Get Wikipedia data
        wikipedia = get_wikipedia()
        raw_wiki_data = wikipedia.run(topic)
        
        # If no Wikipedia data found, return a helpful message
        if not raw_wiki_data or len(raw_wiki_data.strip()) < 10:
            return f"I couldn't find specific information about '{topic}' on Wikipedia. Please try a different search term or topic."
        
        # Create a research prompt
        research_prompt = f"""
You are a research assistant tasked with providing comprehensive information about a topic.
Based on the information provided, create a well-structured research summary that includes:

1. Overview: A brief introduction to the topic
2. Key Facts: Important information and data points
3. History: Relevant historical context and timeline
4. Significance: Why this topic matters
5. Applications: How this information is used in the real world
6. Related Topics: Connections to other fields or concepts

Make your response informative, accurate, and well-organized. If certain information is not available, acknowledge the gaps rather than making assumptions.

Topic: {topic}

Information from Wikipedia:
{raw_wiki_data}

Research Summary:
"""
        
        # Call OpenRouter with the research prompt
        system_prompt = "You are a research assistant tasked with providing comprehensive information about topics."
        return call_openrouter(research_prompt, system_prompt)
    except Exception as e:
        logger.error(f"Error in research function: {str(e)}")
        return f"I encountered an error while researching '{topic}': {str(e)}"

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello! I'm your research assistant. I can help you find information about any topic. What would you like to learn about?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Track input tokens
    input_tokens = count_tokens(message.content)
    cost_tracker.track_usage(input_tokens)
    
    # Process the query
    response = research_topic(message.content)
    
    # Track output tokens
    output_tokens = count_tokens(response)
    cost_tracker.track_usage(output_tokens)
    
    # Send response with usage info
    await cl.Message(
        content=f"{response}\n\n---\nUsage: {input_tokens + output_tokens} tokens (${(input_tokens + output_tokens)/1000*0.002:.4f})"
    ).send() 