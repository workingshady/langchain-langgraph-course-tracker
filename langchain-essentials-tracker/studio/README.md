# LangGraph Studio Experiments

This folder contains experiment files for running agents in LangGraph Studio.

## Available Experiments

### SQL Agent Experiment (`sql_agent_experiment.py`)

An interactive SQL agent that can query the Chinook database using natural language.

**Features:**
- Natural language to SQL query conversion
- Read-only database access (prevents destructive operations)
- Powered by Google Gemini 2.5 Flash
- Streams responses in real-time

**Database:** Chinook.db (Music store database with artists, albums, tracks, customers, invoices)

## Setup

1. **Install Dependencies:**
   ```bash
   uv sync
   ```

2. **Set up Environment Variables:**
   Create a `.env` file in the project root with:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

3. **Run with LangGraph Studio:**
   ```bash
   langgraph dev
   ```

   Then open LangGraph Studio and select the `sql_agent_experiment` graph.

## Running Experiments

### Using LangGraph Studio (Recommended)
1. Start LangGraph Studio: `langgraph dev`
2. Select the `sql_agent_experiment` graph
3. Enter your natural language question
4. Watch the agent reason and execute SQL queries

### Command Line
```bash
python studio/sql_agent_experiment.py
```

### Programmatic Usage
```python
from studio.sql_agent_experiment import run_agent

run_agent("Which table has the largest number of entries?")
```

## Example Questions

- "Which table has the largest number of entries?"
- "Show me the top 5 artists by number of albums"
- "What are the total sales for each country?"
- "List the 5 longest tracks"
- "Which genre has the most tracks?"

## Configuration

The experiment is configured via `langgraph.json`:
- **Dependencies:** Points to the studio folder
- **Graphs:** Maps graph names to Python modules
- **Environment:** Uses `.env` file for secrets

## Troubleshooting

**Error: GOOGLE_API_KEY not set**
- Make sure you have a `.env` file in the project root with your API key

**Database not found**
- The database path is: `data/Chinook.db`
- Make sure the database file exists at this location

**Import errors**
- Run `uv sync` to install all dependencies
- Make sure you're running from the project root

## Architecture

```
┌─────────────────────────────────────────┐
│         LangGraph Studio UI             │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    sql_agent_experiment.py              │
│    - Loads DB connection                │
│    - Builds agent with tools            │
│    - Manages RuntimeContext             │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│        agents/sql-agent/src/            │
│    - agent/sql_agent.py (build_agent)   │
│    - agent/tools.py (execute_sql)       │
│    - agent/prompts.py (SYSTEM_PROMPT)   │
│    - db/connection.py (get_db)          │
│    - context/runtime_context.py         │
└─────────────────────────────────────────┘
```

## Adding New Experiments

1. Create a new Python file in the `studio/` folder
2. Define your agent and expose it as `agent` variable
3. Update `langgraph.json` to include your new graph:
   ```json
   {
     "graphs": {
       "your_experiment": "./studio/your_file.py:agent"
     }
   }
   ```
4. Run `langgraph dev` to test

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio)
- [Chinook Database Schema](https://github.com/lerocha/chinook-database)

