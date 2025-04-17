# Research Agent

The Research Agent is a specialized component of the Multi-Agent Chatbot system designed to provide comprehensive information on various topics. It uses Wikipedia and other sources to gather and structure information in a research-style format.

## Features

- **Structured Research Output**: Provides information in a well-organized format with sections for overview, key facts, history, significance, applications, and related topics.
- **Wikipedia Integration**: Uses Wikipedia as a primary source for factual information.
- **Error Handling**: Gracefully handles cases where information isn't found or errors occur.
- **Cost Tracking**: Tracks token usage and estimated costs for API calls.
- **Multiple Interfaces**: Available both as a standalone CLI tool and integrated into the multi-agent system.

## Usage

### As a Standalone Agent

You can use the Research Agent directly through the Streamlit interface by selecting "Research Agent" from the agent selection dropdown in the sidebar.

### Command Line Interface

The Research Agent can also be used from the command line:

```bash
# Run in interactive mode
python agents/research_agent_cli.py --interactive

# Run with a specific query
python agents/research_agent_cli.py --query "Tell me about quantum computing"
```

### In the Multi-Agent System

The Research Agent is automatically integrated into the multi-agent system. When you ask questions that require research (containing keywords like "research", "find information about", "tell me about", etc.), the system will route your query to the Research Agent.

## Example Queries

- "Research the history of artificial intelligence"
- "Tell me about quantum computing"
- "What is the significance of the Turing machine?"
- "Who is Alan Turing and what were his contributions to computer science?"
- "When did the first computer virus appear?"

## Configuration

The Research Agent uses the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: (Optional) Custom base URL for OpenAI API

These can be set in your `.env` file or through the Streamlit interface.

## Integration with Other Agents

The Research Agent is designed to work seamlessly with other agents in the system:

1. The Planner agent determines if a query requires research
2. The Research Agent gathers and structures the information
3. The Replanner agent evaluates if the research is complete or needs additional steps

## Limitations

- The Research Agent primarily relies on Wikipedia for information, which may not cover all topics in depth
- Some queries may require multiple research steps to gather comprehensive information
- The agent may not always recognize when a query requires research vs. other types of processing 