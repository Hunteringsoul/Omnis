import streamlit as st
import os
from dotenv import load_dotenv
import logging
from agents.research_agent_cli import process_query as research_process_query
from agents.executor import get_executor
from agents.planner import get_planner, get_replanner, get_answerer
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
    
    # Research queries
    if any(keyword in query_lower for keyword in [
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
    
    # Initialize agents if not already initialized
    if "agents_initialized" not in st.session_state:
        executor, planner, replanner, answerer = initialize_agents()
        st.session_state["executor"] = executor
        st.session_state["planner"] = planner
        st.session_state["replanner"] = replanner
        st.session_state["answerer"] = answerer
        st.session_state["agents_initialized"] = True
    
    # Process query with the appropriate agent
    if agent_type == "research":
        response = research_process_query(query)
        output_tokens = count_tokens(response)
        cost_tracker.track_usage(output_tokens)
        return response, input_tokens + output_tokens
    
    elif agent_type == "planner":
        plan_result = st.session_state["planner"]({"objective": query})
        response = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan_result.steps)])
        output_tokens = count_tokens(response)
        cost_tracker.track_usage(output_tokens)
        return response, input_tokens + output_tokens
    
    elif agent_type == "multi-agent":
        # Initialize state
        state = {
            "input": query,
            "plan": [],
            "past_steps": [],
            "response": ""
        }
        
        # Get initial plan
        plan_result = st.session_state["planner"]({"objective": query})
        state["plan"] = plan_result.steps
        
        # Execute steps until we get a response
        while state["plan"] and not state["response"]:
            # Execute the first step
            task = state["plan"][0]
            executor_result = st.session_state["executor"]({"input": task})
            state["past_steps"].append((task, executor_result["output"]))
            
            # Remove the executed step
            state["plan"] = state["plan"][1:]
            
            # Replan or get response
            replan_result = st.session_state["replanner"](state)
            if hasattr(replan_result, "response"):
                state["response"] = replan_result.response
            else:
                state["plan"] = replan_result.steps
        
        # If we still don't have a response, use the answerer
        if not state["response"]:
            answer_result = st.session_state["answerer"]({"input": query})
            state["response"] = answer_result.response
        
        output_tokens = count_tokens(state["response"])
        cost_tracker.track_usage(output_tokens)
        return state["response"], input_tokens + output_tokens
    
    else:  # answerer
        answer_result = st.session_state["answerer"]({"input": query})
        response = answer_result.response
        output_tokens = count_tokens(response)
        cost_tracker.track_usage(output_tokens)
        return response, input_tokens + output_tokens

# Streamlit UI
with st.sidebar:
    st.sidebar.title("API Keys")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", key="openrouter_api_key", type="password")
    
    st.sidebar.title("Agent Selection")
    agent_selection = st.sidebar.radio(
        "Select Agent",
        ["Auto-Detect", "Research Agent", "Planner Agent", "Multi-Agent System", "Direct Answerer"],
        index=0
    )
    
    st.sidebar.title("Usage Statistics")
    st.sidebar.text(cost_tracker.get_usage_summary())

st.title("💬Multi-Agent Search Engine Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key and not openrouter_api_key:
        st.info("Please add your OpenAI API Key or OpenRouter API Key to continue.")
        st.stop()
    
    # Set environment variables for API keys
    if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
    if openrouter_api_key:
        os.environ['OPENROUTER_API_KEY'] = openrouter_api_key
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Determine which agent to use
    agent_type = None
    if agent_selection == "Auto-Detect":
        agent_type = determine_agent(prompt)
    elif agent_selection == "Research Agent":
        agent_type = "research"
    elif agent_selection == "Planner Agent":
        agent_type = "planner"
    elif agent_selection == "Multi-Agent System":
        agent_type = "multi-agent"
    elif agent_selection == "Direct Answerer":
        agent_type = "answerer"
    
    # Process the query with the appropriate agent
    with st.spinner("Processing..."):
        response, total_tokens = process_query(prompt, agent_type)
        
        # Add usage information to the response
        usage_info = f"\n\n---\n*Usage: {total_tokens} tokens (${total_tokens/1000*0.002:.4f})*"
        if agent_type == "Auto-Detect":
            usage_info += f"\n*Agent used: {agent_type}*"
        
        msg = response + usage_info
    
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)



