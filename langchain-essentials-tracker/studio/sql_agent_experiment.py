"""
SQL Agent Experiment for LangGraph Studio

This module sets up the SQL agent for experimentation in LangGraph Studio.
It uses the sql-agent implementation from the agents/sql-agent directory.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the sql-agent directory to the Python path
sql_agent_path = Path(__file__).parent.parent / "agents" / "sql-agent"
sys.path.insert(0, str(sql_agent_path))

from src.agent.sql_agent import build_agent
from src.context.runtime_context import RuntimeContext
from src.db.connection import get_db

# Load environment variables
load_dotenv()

# Get API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in .env file")

# Set up database path (relative to the project root)
DB_PATH = str(Path(__file__).parent.parent / "data" / "Chinook.db")

# Initialize database connection
db = get_db(DB_PATH)

# Build the agent
agent = build_agent(GOOGLE_API_KEY)



def run_agent(question: str):
    """
    Run the SQL agent with a question.

    Args:
        question: Natural language question to ask the database

    Returns:
        The agent's response
    """
    results = []
    for step in agent.stream(
        {"messages": question},
        context=RuntimeContext(db=db),
        stream_mode="values"
    ):
        results.append(step)
        step["messages"][-1].pretty_print()
    return results


# Example usage and test
if __name__ == "__main__":
    print("=" * 80)
    print("SQL Agent Experiment - LangGraph Studio")
    print("=" * 80)
    print(f"\nDatabase: {DB_PATH}")
    print(f"Model: gemini-2.5-flash")
    print("\n" + "=" * 80)

    # Test question
    test_question = "Which table has the largest number of entries?"
    print(f"\nTest Question: {test_question}\n")
    print("=" * 80 + "\n")

    run_agent(test_question)

