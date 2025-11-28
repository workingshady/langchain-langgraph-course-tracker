# LangChain Essentials – Tracking Repository

<p align="center">
  <a href="https://www.langchain.com/" target="_blank" rel="noopener noreferrer">
    <img src="https://theaiinsider.tech/wp-content/uploads/2025/07/Screenshot-3112.png" alt="LangChain Logo" width="700"/>
  </a>
</p>


Welcome to the LangChain Essentials Tracker. This repository documents progress through **LangChain Academy's "LangChain Essentials (Python)"** course. This course covers the core components of LangChain using `create_agent`, a robust and extensible agent framework. Learn how to leverage messages, tools, MCP, streaming, structured outputs, and more to build advanced agents.

## Course Lessons

This repository includes nine notebooks, each focused on a fundamental aspect of LangChain, beginning with the new Create Agent workflow.

### L1_fast_agent.ipynb - Create Agent
Utilize LangChain's `create_agent` to build an SQL agent with minimal code. Experience how simple it is to construct powerful agents and use LangSmith Studio for visual debugging and agent exploration.

### L2-7 - Building Blocks
Explore LangChain's fundamental building blocks:

- **L2_messages.ipynb**: How messages facilitate communication between agent components
- **L3_streaming.ipynb**: Enhance responsiveness by implementing streaming
- **L4_tools.ipynb**: Leverage custom or prebuilt tools to extend model capabilities
- **L5_tools_with_mcp.ipynb**: Integrate with the LangChain MCP adapter for MCP tools
- **L6_memory.ipynb**: Enable agents to maintain and utilize state across interactions
- **L7_structuredOutput.ipynb**: Generate and manage structured outputs from your agent

### L8-9 - Agent Customization
`create_agent` enables both prebuilt and user-defined customization through middleware:

- **L8_dynamic.ipynb**: Dynamically alter the agent's system prompt in response to changing contexts
- **L9_HITL.ipynb**: Implement Human-in-the-Loop (HITL) using Interrupts for enhanced interactivity

## Project Structure

```
langchain-essentials-tracker/
│
├── README.md
├── .gitignore
├── pyproject.toml
├── uv.lock
├── .env
│
├── notebooks/            # Jupyter notebooks for lessons
│   ├── L1_fast_agent.ipynb
│   ├── L2_messages.ipynb
│   ├── L3_streaming.ipynb
│   ├── L4_tools.ipynb
│   ├── L5_tools_with_mcp.ipynb
│   ├── L6_memory.ipynb
│   ├── L7_structuredOutput.ipynb
│   ├── L8_dynamic.ipynb
│   └── L9_HITL.ipynb
│
├── examples/             # Python examples
│   ├── simple_sql_agent.py
│   ├── custom_tools.py
│   ├── streaming_demo.py
│   └── structured_output_example.py
│
├── studio/               # LangSmith studio files
│
│
└── agents/                  # Source code
    ├── __init__.py


```

## Quick Start

### 1. Set Up Environment

Using `uv` (recommended):

```bash
uv sync
```

Create a `.env` file and provide your API keys:

```
OPENAI_API_KEY=your_openai_key_here
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-py-essentials
```

## Key Dependencies

- `langchain` - Core LangChain library
- `langchain-community` - Community integrations
- `langchain-openai` - OpenAI integrations
- `langchain-core` - Core abstractions
- `langsmith` - LangSmith tracing and monitoring
- `langgraph` - Graph-based agent framework
- `jupyterlab` - Interactive notebook environment
- `python-dotenv` - Manage environment variables
- `pandas` & `numpy` - Data processing libraries

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Course Repository](https://github.com/langchain-ai/lca-langchainV1-essentials)

## License

This is a personal learning repository based on [LangChain Academy's LangChain Essentials course](https://github.com/langchain-ai/lca-langchainV1-essentials).

---
