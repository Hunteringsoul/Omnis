from flask import Flask, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv
import logging
from agents.research_agent_cli import process_query as research_process_query
from agents.executor import get_executor
from agents.planner import get_planner, get_replanner, get_answerer
from agents.coding_agent_cli import generate_code, explain_code, debug_code
from agents.concept_chart_agent import process_query as concept_chart_process_query
from agents.response_formatter import format_structured_response
import tiktoken
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='.')

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
    query = query.lower()
    
    # Check for visualization requests first (most specific)
    if any(word in query for word in ["plot", "chart", "visualize", "compare", "graph", "show", "display", "bar", "line", "pie"]):
        return "concept_chart"
    
    # Check for concept mapping requests
    if any(word in query for word in ["concept", "map", "outline", "structure"]):
        return "concept_chart"
    
    # Check for coding-related queries
    if any(word in query for word in ["code", "program", "function", "class", "debug", "error", "fix", "implement"]):
        return "coding"
    
    # Check for mathematical queries
    if any(word in query for word in ["calculate", "math", "equation", "formula", "solve", "compute"]):
        return "math"
    
    # Check for research-related queries
    if any(word in query for word in ["research", "find", "search", "look up", "information about", "what is", "who is", "where is"]):
        return "research"
    
    # Check for planning-related queries
    if any(word in query for word in ["plan", "strategy", "approach", "steps", "how to", "guide", "tutorial"]):
        return "planner"
    
    # Default to multi-agent for complex queries
    return "multi_agent"

# Process query with the appropriate agent
def process_query(query, agent_type=None):
    # Track input tokens
    input_tokens = count_tokens(query)
    cost_tracker.track_usage(input_tokens)
    
    # Determine agent type if not specified
    if not agent_type:
        agent_type = determine_agent(query)
    
    try:
        if agent_type == "concept_chart":
            result = concept_chart_process_query(query)
            if result["type"] == "chart":
                # For chart responses, we need to handle the image file
                return {
                    "response": format_structured_response(result["message"], agent_type, query),
                    "image_path": result["image_path"],
                    "agent": "concept_chart",
                    "usage": {
                        "tokens": input_tokens,
                        "cost": input_tokens * 0.000002
                    }
                }
            else:
                return {
                    "response": format_structured_response(result["message"], agent_type, query),
                    "agent": "concept_chart",
                    "usage": {
                        "tokens": input_tokens,
                        "cost": input_tokens * 0.000002
                    }
                }
        elif agent_type == "coding":
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
            
            return {
                "response": response,
                "agent": "coding",
                "usage": {
                    "tokens": input_tokens + output_tokens,
                    "cost": (input_tokens + output_tokens) * 0.000002
                }
            }
        
        elif agent_type == "math":
            # Process with math agent
            response = math_process_query(query)
            output_tokens = count_tokens(response)
            cost_tracker.track_usage(output_tokens)
            
            return {
                "response": format_structured_response(response, agent_type, query),
                "agent": "math",
                "usage": {
                    "tokens": input_tokens + output_tokens,
                    "cost": (input_tokens + output_tokens) * 0.000002
                }
            }
        
        elif agent_type == "research":
            # Process with research agent
            response = research_process_query(query)
            output_tokens = count_tokens(response)
            cost_tracker.track_usage(output_tokens)
            
            return {
                "response": format_structured_response(response, agent_type, query),
                "agent": "research",
                "usage": {
                    "tokens": input_tokens + output_tokens,
                    "cost": (input_tokens + output_tokens) * 0.000002
                }
            }
        
        elif agent_type == "planner":
            # Process with planner agent
            response = planner_process_query(query)
            output_tokens = count_tokens(response)
            cost_tracker.track_usage(output_tokens)
            
            return {
                "response": format_structured_response(response, agent_type, query),
                "agent": "planner",
                "usage": {
                    "tokens": input_tokens + output_tokens,
                    "cost": (input_tokens + output_tokens) * 0.000002
                }
            }
        
        elif agent_type == "multi_agent":
            # Process with multi-agent system
            response = multi_agent_process_query(query)
            output_tokens = count_tokens(response)
            cost_tracker.track_usage(output_tokens)
            
            return {
                "response": format_structured_response(response, agent_type, query),
                "agent": "multi_agent",
                "usage": {
                    "tokens": input_tokens + output_tokens,
                    "cost": (input_tokens + output_tokens) * 0.000002
                }
            }
        
        else:
            # Default to answerer agent
            response = answerer_process_query(query)
            output_tokens = count_tokens(response)
            cost_tracker.track_usage(output_tokens)
            
            return {
                "response": format_structured_response(response, agent_type, query),
                "agent": "answerer",
                "usage": {
                    "tokens": input_tokens + output_tokens,
                    "cost": (input_tokens + output_tokens) * 0.000002
                }
            }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "response": f"An error occurred: {str(e)}",
            "agent": agent_type,
            "usage": {
                "tokens": input_tokens,
                "cost": input_tokens * 0.000002
            }
        }

# Routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def styles():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        agent_type = data.get('agent', 'auto')
        
        # Process the query
        result = process_query(message, agent_type)
        
        # Prepare the response
        response = {
            "response": result["response"],
            "agent": result["agent"],
            "usage": result["usage"]
        }
        
        # If there's an image path, add it to the response
        if "image_path" in result:
            response["image_path"] = result["image_path"]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            "error": "An error occurred while processing your request.",
            "details": str(e)
        }), 500

@app.route('/api/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('.', filename)

@app.route('/api/usage', methods=['GET'])
def usage():
    return jsonify({
        'total_tokens': cost_tracker.total_tokens,
        'total_cost': cost_tracker.total_cost,
        'daily_usage': cost_tracker.daily_usage
    })

# Process functions for different agent types
def math_process_query(query):
    executor = get_executor()
    result = executor({"input": query})
    return result["output"]

def planner_process_query(query):
    planner = get_planner()
    plan_result = planner({"objective": query})
    return "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan_result.steps)])

def multi_agent_process_query(query):
    # Initialize agents
    executor = get_executor()
    planner = get_planner()
    replanner = get_replanner()
    answerer = get_answerer()
    
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
    
    return state["response"]

def answerer_process_query(query):
    answerer = get_answerer()
    answer_result = answerer({"input": query})
    return answer_result.response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("Multi-Agent Chatbot Web Interface")
    print("=================================")
    print("Initializing agents...")
    executor, planner, replanner, answerer = initialize_agents()
    print("Agents initialized successfully!")
    print(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")
    print(f"Starting web server on port: {port}")
    app.run(host="0.0.0.0", port=port, debug=False if os.environ.get("PORT") else True)
