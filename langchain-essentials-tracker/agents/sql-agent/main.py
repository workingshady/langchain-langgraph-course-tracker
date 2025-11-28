import argparse
from dotenv import load_dotenv
import os
from src.db.connection import get_db
from src.agent.sql_agent import build_agent
from src.context.runtime_context import RuntimeContext

def main():
    parser = argparse.ArgumentParser(description="Interact with the SQLite SQL Agent.")
    parser.add_argument(
        "-q", "--question",
        type=str,
        default="Which table has the largest number of entries?",
        help="The natural language question to ask the SQL agent."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/Chinook.db",
        help="Path to the SQLite database file."
    )
    args = parser.parse_args()

    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY not set")

    db = get_db(args.db_path)
    agent = build_agent(GOOGLE_API_KEY)

    for step in agent.stream({"messages": args.question}, context=RuntimeContext(db=db), stream_mode="values"):
        step["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()
