from typing import List, Tuple, Annotated, TypedDict
import operator
from agents.planner import get_planner, get_replanner
from agents.executor import get_executor
from agents.research_agent import setup_research_agent
from agents.planner import Response
from langgraph.graph import StateGraph, END

import os
from uuid import uuid4
from langsmith import Client

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-search"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()
# Define the State
class State(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    

# Create the graph

def get_graph():
    # Create agents
    executor = get_executor()
    planner = get_planner()
    replanner = get_replanner()
    research_agent = setup_research_agent()

    def execute(state: State):
        task = state['plan'][0]
        output = executor.invoke({'input': task, "chat_history" : []})
        return {"past_steps" : (task, output['agent_outcome'].return_values['output'])}

    def plan(state: State):
        plan = planner.invoke({'objective': state['input']})
        return {"plan" : plan.steps}

    def replan(state: State):
        output = replanner.invoke(state)
        # If the output is a response (the plan is complete), then return the response else update the plan
        if isinstance(output, Response):
            return {"response" : output.response}
        else:
            return {"plan" : output.steps}
            
    def research(state: State):
        # Check if the task is a research query
        task = state['plan'][0]
        if any(keyword in task.lower() for keyword in ['research', 'find information about', 'tell me about', 'what is', 'who is', 'when did', 'where is', 'how does']):
            output = research_agent.invoke({'input': task})
            return {"past_steps": (task, output['output'])}
        else:
            # If not a research query, pass to the executor
            return execute(state)

    def should_end(state: State):
        if (state['response']):
            return True
        else:
            return False
        
    graph = StateGraph(State)
    graph.add_node("planner", plan)
    graph.add_node("executor", execute)
    graph.add_node("replanner", replan)
    graph.add_node("research", research)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "replanner")
    graph.add_conditional_edges(
        'replanner',
        should_end,
        {
            True: END,
            False: "research"
        })
    return graph.compile()



