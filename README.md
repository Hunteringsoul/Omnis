# Multi-Agent Chatbot System

A powerful multi-agent chatbot system that can handle various types of queries using specialized agents.

## Features

- **Multiple Specialized Agents**:
  - **Coding Agent**: Generates, explains, and debugs code
  - **Math Agent**: Handles mathematical calculations and concepts
  - **Research Agent**: Provides information from Wikipedia and other sources
  - **Planner Agent**: Creates step-by-step plans for tasks
  - **Multi-Agent**: Combines multiple agents for complex tasks
  - **Answerer Agent**: Provides direct answers to simple questions

- **Two Interfaces**:
  - **CLI Interface**: Command-line interface for direct interaction
  - **Web Interface**: Modern web UI with a beautiful design

- **Token Usage Tracking**: Monitors and reports token usage and costs

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multi-agent-chatbot.git
   cd multi-agent-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Usage

### CLI Interface

Run the CLI interface:
```
python cli.py
```

Available commands:
- Type your question directly to use auto-detection
- `auto <question>` - Use auto-detection for agent selection
- `coding <question>` - Use the coding agent
- `executor <question>` - Use the math agent
- `research <question>` - Use the research agent
- `planner <question>` - Use the planner agent
- `multi <question>` - Use the multi-agent system
- `answerer <question>` - Use the direct answerer
- `usage` - Show usage statistics
- `help` - Show available commands
- `exit` - Exit the application

### Web Interface

Run the web interface:
```
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

The web interface provides:
- A modern, responsive design
- Agent selection via a dropdown menu
- Real-time chat interaction
- Token usage tracking
- Beautiful animations and effects

## Agent Capabilities

### Coding Agent
- Generates code in various programming languages
- Explains existing code
- Debugs code with error messages

### Math Agent
- Performs mathematical calculations
- Explains mathematical concepts
- Handles topics like prime numbers, Armstrong numbers, etc.

### Research Agent
- Searches Wikipedia for information
- Provides comprehensive summaries
- Cites sources

### Planner Agent
- Creates step-by-step plans for tasks
- Breaks down complex tasks into manageable steps

### Multi-Agent
- Combines multiple agents for complex tasks
- Dynamically plans and executes steps
- Provides comprehensive responses

### Answerer Agent
- Provides direct answers to simple questions
- Handles general queries

## License

MIT
