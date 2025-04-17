import chainlit as cl
from langchain_openai import OpenAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
import tiktoken
from functools import lru_cache
import json
from datetime import datetime
import os

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

# Initialize cost tracker
cost_tracker = CostTracker()

@cl.on_chat_start
def math_chatbot():
    # Initialize OpenAI with cost-efficient settings
    llm = OpenAI(
        model='gpt-3.5-turbo-instruct',
        temperature=0,
        max_tokens=500  # Limit response length to save tokens
    )

    # Cached Wikipedia wrapper
    @lru_cache(maxsize=1)
    def get_wikipedia():
        return WikipediaAPIWrapper()

    word_problem_template = """You are a reasoning agent tasked with solving the user's logic-based questions.
    Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved and give
    the final answer. Provide the response in bullet points. Question: {question} Answer:"""

    math_assistant_prompt = PromptTemplate(
        input_variables=["question"],
        template=word_problem_template
    )

    # Initialize chains
    word_problem_chain = LLMChain(llm=llm, prompt=math_assistant_prompt)
    problem_chain = LLMMathChain.from_llm(llm=llm)

    # Initialize tools
    word_problem_tool = Tool.from_function(
        name="Reasoning Tool",
        func=word_problem_chain.run,
        description="Useful for when you need to answer logic-based/reasoning questions."
    )

    math_tool = Tool.from_function(
        name="Calculator",
        func=problem_chain.run,
        description="Useful for when you need to answer numeric questions. This tool is only for math questions and nothing else. Only input math expressions, without text"
    )

    wikipedia_tool = Tool(
        name="Wikipedia",
        func=get_wikipedia().run,
        description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions."
    )

    # Initialize agent
    agent = initialize_agent(
        tools=[wikipedia_tool, math_tool, word_problem_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True
    )

    # Store agent and cost tracker in session
    cl.user_session.set("agent", agent)
    cl.user_session.set("cost_tracker", cost_tracker)

@cl.on_message
async def process_user_query(message: cl.Message):
    agent = cl.user_session.get("agent")
    cost_tracker = cl.user_session.get("cost_tracker")

    # Track input tokens
    input_tokens = count_tokens(message.content)
    cost_tracker.track_usage(input_tokens)

    # Process the query
    response = await agent.acall(
        message.content,
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    # Track output tokens
    output_tokens = count_tokens(response["output"])
    cost_tracker.track_usage(output_tokens)

    # Create response message with usage info
    usage_info = f"\n\n---\n*Usage: {input_tokens + output_tokens} tokens (${(input_tokens + output_tokens)/1000*0.002:.4f})*"
    
    await cl.Message(
        content=response["output"] + usage_info
    ).send() 