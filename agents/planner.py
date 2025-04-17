import os
from openai import OpenAI
from pydantic import BaseModel, Field
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

class Planner(BaseModel):
    """Plan the execution of the agent"""
    
    steps: list[str] = Field(
        description="The steps to execute the agent, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user"""
    
    response: str
    
def call_openrouter(prompt, system_prompt="You are a helpful assistant that can create plans and provide responses."):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Increased temperature for more varied responses
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenRouter: {str(e)}")
        return f"Error: {str(e)}"

def get_planner():
    """
    Returns a function that can be used to create a plan for a given objective.
    """
    def planner(objective_dict):
        objective = objective_dict.get("objective", "")
        
        # Create a prompt for the planner
        prompt = f"""
Given the objective, devise a simple and concise step-by-step plan that involves using a search engine to find the answer. 
This plan should consist of individual search queries that, if executed correctly, will yield the correct answer. 
Avoid unnecessary steps and aim to make the plan as short as possible. The result of the final step should be the final answer.

Objective: {objective}

Please provide a natural, conversational response with the steps. Don't use a rigid format or numbering unless it makes sense for the specific query.
"""
        
        # Call OpenRouter with the prompt
        system_prompt = "You are a helpful assistant that creates natural, conversational plans. Your response should be helpful and easy to follow, but not rigidly structured."
        response = call_openrouter(prompt, system_prompt)
        
        # Extract steps from the response
        try:
            # Try to identify steps in the response
            lines = response.strip().split('\n')
            steps = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for numbered or bulleted lists
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.', 
                                   '-', '*', '•', '→', '>>', '>', '•')):
                    # Remove numbering/bullets
                    if '. ' in line:
                        line = line.split('. ', 1)[1]
                    elif ' ' in line:
                        line = line.split(' ', 1)[1]
                    steps.append(line)
                # Check for sentences that might be steps
                elif line.endswith('.') and len(line) < 100:
                    steps.append(line)
            
            # If no steps were found, treat the entire response as one step
            if not steps:
                steps = [response]
            
            return Planner(steps=steps)
        except Exception as e:
            logger.error(f"Error parsing planner response: {str(e)}")
            # Return a simple plan if parsing fails
            return Planner(steps=[f"Search for information about: {objective}"])
    
    return planner

def get_replanner():
    """
    Returns a function that can be used to update a plan or return a response.
    """
    def replanner(state_dict):
        input_text = state_dict.get("input", "")
        plan = state_dict.get("plan", "")
        past_steps = state_dict.get("past_steps", "")
        
        # Create a prompt for the replanner
        prompt = f"""
Your task is to revise the current plan based on the executed steps. Remove any steps that have been completed and ensure the remaining steps will lead to the complete answer for the objective. Remember, the objective should be fully answered, not just partially. If the answer is already found in the executed steps, return the answer to the objective.

Objective:
{input_text}

Current Plan:
{plan}

Executed Steps:
{past_steps}

Please provide a natural, conversational response. If you're providing a final answer, give it directly. If you're updating the plan, explain what steps remain in a natural way.
"""
        
        # Call OpenRouter with the prompt
        system_prompt = "You are a helpful assistant that revises plans or provides final answers in a natural, conversational way."
        response = call_openrouter(prompt, system_prompt)
        
        # Determine if this is a response or updated plan
        try:
            # Check if the response looks like a final answer (short, direct)
            if len(response.split('\n')) < 3 and len(response) < 500:
                return Response(response=response)
            
            # Otherwise, treat as an updated plan
            lines = response.strip().split('\n')
            steps = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for numbered or bulleted lists
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.', 
                                   '-', '*', '•', '→', '>>', '>', '•')):
                    # Remove numbering/bullets
                    if '. ' in line:
                        line = line.split('. ', 1)[1]
                    elif ' ' in line:
                        line = line.split(' ', 1)[1]
                    steps.append(line)
                # Check for sentences that might be steps
                elif line.endswith('.') and len(line) < 100:
                    steps.append(line)
            
            # If no steps were found, treat the entire response as one step
            if not steps:
                steps = [response]
            
            return Planner(steps=steps)
        except Exception as e:
            logger.error(f"Error parsing replanner response: {str(e)}")
            # Return a simple response if parsing fails
            return Response(response=f"I encountered an error while processing your request: {str(e)}")
    
    return replanner

def get_answerer():
    """
    Returns a function that can be used to answer any question directly.
    """
    def answerer(query_dict):
        query = query_dict.get("input", "")
        
        # Create a prompt for the answerer
        prompt = f"""
Please provide a comprehensive answer to the following question or request.
If the question requires factual information, provide accurate details.
If the question is about a task or process, provide step-by-step instructions.
If the question is asking for an opinion or analysis, provide a well-reasoned response.

Question: {query}

Please provide a detailed and helpful response in a natural, conversational tone. Adapt your response style to match the query - be formal for serious topics, friendly for casual questions, etc.
"""
        
        # Call OpenRouter with the prompt
        system_prompt = "You are a helpful assistant that provides comprehensive answers in a natural, conversational tone. Adapt your response style to match the query."
        response = call_openrouter(prompt, system_prompt)
        
        return Response(response=response)
    
    return answerer

# Testing
if __name__ == "__main__":
    print("Planner Agent CLI")
    print("=================")
    print("Type 'exit' to quit")
    print("Ask any question or request a plan!")
    print("===============================================\n")
    
    planner = get_planner()
    replanner = get_replanner()
    answerer = get_answerer()
    
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
            
            # Determine if this is a planning request or a direct question
            if any(keyword in user_input.lower() for keyword in ["how to", "steps", "plan", "process", "guide", "instructions"]):
                # Get the plan
                plan_result = planner({"objective": user_input})
                print("\nPlan:")
                # Print the steps in a more natural way
                for i, step in enumerate(plan_result.steps, 1):
                    print(f"{i}. {step}")
                
                # Track output tokens
                output_tokens = count_tokens("\n".join(plan_result.steps))
            else:
                # Get direct answer
                answer_result = answerer({"input": user_input})
                print(f"\nAnswer: {answer_result.response}")
                
                # Track output tokens
                output_tokens = count_tokens(answer_result.response)
            
            cost_tracker.track_usage(output_tokens)
            print(f"\nUsage: {input_tokens + output_tokens} tokens (${(input_tokens + output_tokens)/1000*0.002:.4f})")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")
    
    