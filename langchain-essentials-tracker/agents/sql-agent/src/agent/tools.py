from langgraph.runtime import get_runtime
from langchain_core.tools import tool
from ..context.runtime_context import RuntimeContext

@tool
def execute_sql(query: str) -> str:
    """
    Execute a SQL query on the SQLite database and return the result as a string.

    Prevents destructive operations such as INSERT, UPDATE, DELETE, ALTER, DROP, CREATE, REPLACE, and TRUNCATE.
    """
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    # Prevent destructive operations
    forbidden = ['INSERT', 'UPDATE', 'DELETE', 'ALTER', 'DROP', 'CREATE', 'REPLACE', 'TRUNCATE']
    if any(keyword in query.upper() for keyword in forbidden):
        return "Error: Destructive operations are not allowed"

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"
