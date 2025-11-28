from langchain_community.utilities import SQLDatabase

def get_db(path: str = "data/Chinook.db") -> SQLDatabase:
    """
    Returns a SQLDatabase instance connected to the specified SQLite database file.

    Args:
        path (str): The path to the SQLite database file. Defaults to "data/Chinook.db".

    Returns:
        SQLDatabase: An instance of SQLDatabase connected to the database file.
    """
    return SQLDatabase.from_uri(f"sqlite:///{path}")
