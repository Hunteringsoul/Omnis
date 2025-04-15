from typing import Dict, List, Tuple, Any
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from math_agent import MathAgent
import json

class ExecutorAgent:
    def __init__(self, openai_api_key: str):
        self.math_agent = MathAgent(openai_api_key)
        self.tools = self._setup_tools()
        self.workflow = self._create_workflow()
        
    def _setup_tools(self) -> List:
        @tool
        def solve_math_problem(problem: str) -> str:
            """Solves a math problem using the math agent."""
            return self.math_agent.solve(problem)
            
        return [solve_math_problem]
        
    def _create_workflow(self):
        # Create a new graph
        workflow = StateGraph()
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._execute_tool)
        
        # Add edges
        workflow.add_edge("agent", "tools")
        workflow.add_edge("tools", "agent")
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        return workflow.compile()
        
    def _agent_node(self, state: Dict) -> Dict:
        """Process the current state and decide next action."""
        messages = state.get("messages", [])
        next_step = state.get("next_step", "agent")
        
        if next_step == "agent":
            # Process with agent logic
            last_message = messages[-1] if messages else None
            if isinstance(last_message, HumanMessage):
                # Check if it's a math problem
                if any(keyword in last_message.content.lower() for keyword in ["solve", "calculate", "math", "equation"]):
                    return {
                        "messages": messages,
                        "next_step": "tools",
                        "tool": "solve_math_problem",
                        "tool_input": last_message.content
                    }
                    
        return {"messages": messages, "next_step": "end"}
        
    def _execute_tool(self, state: Dict) -> Dict:
        """Execute the selected tool."""
        tool_name = state.get("tool")
        tool_input = state.get("tool_input")
        messages = state.get("messages", [])
        
        if tool_name == "solve_math_problem":
            result = self.tools[0](tool_input)
            messages.append(AIMessage(content=result))
            
        return {"messages": messages, "next_step": "agent"}
        
    def process(self, message: str) -> str:
        """Process a message through the workflow."""
        state = {
            "messages": [HumanMessage(content=message)],
            "next_step": "agent"
        }
        
        result = self.workflow.invoke(state)
        return result["messages"][-1].content if result["messages"] else "No response generated."

if __name__ == "__main__":
    # Example usage
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    executor = ExecutorAgent(api_key)
    
    # Test with a math problem
    response = executor.process("Can you solve this equation: 2x + 5 = 13?")
    print(response) 