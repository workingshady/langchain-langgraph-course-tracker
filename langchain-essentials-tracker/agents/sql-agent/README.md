# SQL Agent

An intelligent SQL agent built with LangChain and LangGraph that can answer natural language questions about a SQLite database using Google's Gemini AI model.

## Overview

This SQL agent translates natural language questions into SQL queries, executes them against the Chinook database, and returns human-readable answers. The agent is designed with safety features to prevent destructive operations and can self-correct when encountering SQL errors.

## Features

- **Natural Language Interface**: Ask questions in plain English about your database
- **Self-Correcting**: Automatically discovers table schemas and corrects SQL errors
- **Safety First**: Prevents destructive operations (INSERT, UPDATE, DELETE, etc.)
- **Streaming Output**: Real-time response streaming for better user experience
- **Smart Query Optimization**: Uses efficient SQL patterns and follows best practices
- **Powered by Gemini 2.5 Flash**: Fast and accurate responses using Google's latest AI model

## Project Structure

```
sql-agent/
├── data/
│   └── Chinook.db          # SQLite database (music store data)
├── src/
│   ├── agent/
│   │   ├── sql_agent.py    # Agent configuration and builder
│   │   ├── tools.py        # SQL execution tool
│   │   └── prompts.py      # System prompts
│   ├── context/
│   │   └── runtime_context.py  # Runtime context for database access
│   └── db/
│       └── connection.py   # Database connection utilities
├── main.py                 # Entry point
└── README.md              # This file
```

## Prerequisites

- Python 3.10+
- Google API Key (for Gemini AI)
- Virtual environment (recommended)

## Installation & Setup

1. **Navigate to the project root** (not the sql-agent directory):
   ```bash
   cd path/to/langchain-essentials-tracker
   ```

2. **Activate your virtual environment**:
   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Ensure dependencies are installed**:
   The required packages should already be installed in your virtual environment. If not:
   ```bash
   pip install langchain langchain-google-genai langchain-community langgraph python-dotenv
   ```

4. **Set up your Google API Key**:
   Create a `.env` file in the project root with:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

Navigate to the sql-agent directory and run the agent:

```bash
cd agents/sql-agent
python main.py -q "Your question here"
```

### Examples

**Example 1: Top customers by purchases**
```bash
python main.py -q "What are the top 5 customers by total purchases?"
```

**Example 2: Database exploration**
```bash
python main.py -q "Which table has the largest number of entries?"
```

**Example 3: Complex queries**
```bash
python main.py -q "Show me the most popular music genres by number of tracks"
```

### Command-Line Options

- `-q, --question`: The natural language question to ask (default: "Which table has the largest number of entries?")
- `--db-path`: Path to the SQLite database file (default: "data/Chinook.db")
