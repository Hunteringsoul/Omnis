import os
from dotenv import load_dotenv
import logging
from agents.research_agent_cli import process_query as research_process_query
from agents.executor import get_executor
from agents.planner import get_planner, get_replanner, get_answerer
from agents.coding_agent_cli import generate_code, explain_code, debug_code
import tiktoken
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

# Initialize agents
def initialize_agents():
    executor = get_executor()
    planner = get_planner()
    replanner = get_replanner()
    answerer = get_answerer()
    return executor, planner, replanner, answerer

# Determine which agent should handle the query
def determine_agent(query):
    query_lower = query.lower()
    
    # Coding/execution queries
    if any(keyword in query_lower for keyword in [
        'code', 'program', 'write', 'implement', 'function', 'class',
        'script', 'debug', 'fix', 'error', 'exception', 'run', 'execute',
        'compile', 'build', 'test', 'algorithm', 'data structure'
    ]):
        return "coding"
    
    # Mathematical concepts and calculations
    elif any(keyword in query_lower for keyword in [
        'calculate', 'math', 'mathematical', 'formula', 'equation', 'solve',
        'prime', 'armstrong', 'fibonacci', 'factorial', 'gcd', 'lcm',
        'square root', 'power', 'exponent', 'logarithm', 'trigonometry',
        'geometry', 'algebra', 'calculus', 'statistics', 'probability'
    ]):
        return "executor"
    
    # Research queries
    elif any(keyword in query_lower for keyword in [
        'research', 'find information about', 'tell me about', 'what is', 
        'who is', 'when did', 'where is', 'how does', 'explain', 'describe',
        'history of', 'meaning of', 'definition of', 'facts about'
    ]):
        return "research"
    
    # Planning queries
    elif any(keyword in query_lower for keyword in [
        'how to', 'steps', 'plan', 'process', 'guide', 'instructions',
        'method', 'way to', 'approach', 'strategy', 'tutorial'
    ]):
        return "planner"
    
    # Multi-agent system queries (complex tasks that need multiple steps)
    elif any(keyword in query_lower for keyword in [
        'compare', 'analyze', 'evaluate', 'investigate', 'study',
        'find out', 'figure out', 'determine', 'calculate', 'solve'
    ]) or len(query.split()) > 15:
        return "multi-agent"
    
    # Simple queries (direct answers)
    else:
        return "answerer"

# Process query with the appropriate agent
def process_query(query, agent_type=None):
    # Track input tokens
    input_tokens = count_tokens(query)
    cost_tracker.track_usage(input_tokens)
    
    # If agent_type is not specified, determine it
    if agent_type is None:
        agent_type = determine_agent(query)
    
    # Initialize agents
    executor, planner, replanner, answerer = initialize_agents()
    
    # Process query with the appropriate agent
    if agent_type == "coding":
        # Determine if it's a code generation, explanation, or debugging request
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['debug', 'fix', 'error', 'exception']):
            # Extract code and error message if present
            parts = query.split(' ', 1)
            if len(parts) > 1:
                code = parts[1]
                error_message = None
                if 'error:' in code:
                    code_parts = code.split('error:', 1)
                    code = code_parts[0].strip()
                    error_message = code_parts[1].strip()
                response = debug_code(code, error_message)
            else:
                response = "Please provide code to debug."
        elif any(keyword in query_lower for keyword in ['explain', 'how does', 'what does']):
            # Extract code to explain
            parts = query.split(' ', 1)
            if len(parts) > 1:
                code = parts[1]
                response = explain_code(code)
            else:
                response = "Please provide code to explain."
        else:
            # Default to code generation
            language = "python"  # Default language
            if "in " in query_lower and any(lang in query_lower for lang in ['python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust']):
                for lang in ['python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust']:
                    if lang in query_lower:
                        language = lang
                        break
            
            response = generate_code(query, language)
        
        output_tokens = count_tokens(response)
        cost_tracker.track_usage(output_tokens)
        return response, input_tokens + output_tokens, agent_type
    
    elif agent_type == "executor":
        executor_result = executor({"input": query})
        response = executor_result["output"]
        output_tokens = count_tokens(response)
        cost_tracker.track_usage(output_tokens)
        return response, input_tokens + output_tokens, agent_type
    
    elif agent_type == "research":
        response = research_process_query(query)
        output_tokens = count_tokens(response)
        cost_tracker.track_usage(output_tokens)
        return response, input_tokens + output_tokens, agent_type
    
    elif agent_type == "planner":
        plan_result = planner({"objective": query})
        response = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan_result.steps)])
        output_tokens = count_tokens(response)
        cost_tracker.track_usage(output_tokens)
        return response, input_tokens + output_tokens, agent_type
    
    elif agent_type == "multi-agent":
        # Initialize state
        state = {
            "input": query,
            "plan": [],
            "past_steps": [],
            "response": ""
        }
        
        # Get initial plan
        plan_result = planner({"objective": query})
        state["plan"] = plan_result.steps
        
        # Execute steps until we get a response
        while state["plan"] and not state["response"]:
            # Execute the first step
            task = state["plan"][0]
            executor_result = executor({"input": task})
            state["past_steps"].append((task, executor_result["output"]))
            
            # Remove the executed step
            state["plan"] = state["plan"][1:]
            
            # Replan or get response
            replan_result = replanner(state)
            if hasattr(replan_result, "response"):
                state["response"] = replan_result.response
            else:
                state["plan"] = replan_result.steps
        
        # If we still don't have a response, use the answerer
        if not state["response"]:
            answer_result = answerer({"input": query})
            state["response"] = answer_result.response
        
        output_tokens = count_tokens(state["response"])
        cost_tracker.track_usage(output_tokens)
        return state["response"], input_tokens + output_tokens, agent_type
    
    else:  # answerer
        answer_result = answerer({"input": query})
        response = answer_result.response
        output_tokens = count_tokens(response)
        cost_tracker.track_usage(output_tokens)
        return response, input_tokens + output_tokens, agent_type

# CLI interface
if __name__ == "__main__":
    print("Multi-Agent Search Engine Chatbot (CLI Version)")
    print("==============================================")
    print("Type 'exit' to quit")
    print("Type 'help' for available commands")
    print("==============================================")
    
    # Initialize agents
    print("\nInitializing agents...")
    executor, planner, replanner, answerer = initialize_agents()
    print("Agents initialized successfully!")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nExiting...")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  exit - Exit the application")
                print("  help - Show this help message")
                print("  usage - Show usage statistics")
                print("  auto - Use auto-detect for agent selection")
                print("  coding - Use the coding agent (for programming tasks)")
                print("  executor - Use the executor agent (for math tasks)")
                print("  research - Use the research agent")
                print("  planner - Use the planner agent")
                print("  multi - Use the multi-agent system")
                print("  answerer - Use the direct answerer")
                continue
            
            if user_input.lower() == 'usage':
                print(cost_tracker.get_usage_summary())
                continue
            
            # Determine which agent to use
            agent_type = None
            if user_input.lower().startswith('auto '):
                agent_type = None
                user_input = user_input[5:].strip()
            elif user_input.lower().startswith('coding '):
                agent_type = "coding"
                user_input = user_input[7:].strip()
            elif user_input.lower().startswith('executor '):
                agent_type = "executor"
                user_input = user_input[9:].strip()
            elif user_input.lower().startswith('research '):
                agent_type = "research"
                user_input = user_input[9:].strip()
            elif user_input.lower().startswith('planner '):
                agent_type = "planner"
                user_input = user_input[8:].strip()
            elif user_input.lower().startswith('multi '):
                agent_type = "multi-agent"
                user_input = user_input[6:].strip()
            elif user_input.lower().startswith('answerer '):
                agent_type = "answerer"
                user_input = user_input[9:].strip()
            
            if not user_input:
                print("Please enter a query.")
                continue
            
            print("\nProcessing your query...")
            
            # Process the query with the appropriate agent
            response, total_tokens, used_agent = process_query(user_input, agent_type)
            
            # Print the response
            print("\nResponse:")
            print(response)
            
            # Print usage information
            print(f"\nUsage: {total_tokens} tokens (${total_tokens/1000*0.002:.4f})")
            print(f"Agent used: {used_agent}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.") 