from dataclasses import dataclass, field
from typing import Optional

from langchain_community.utilities import SQLDatabase


@dataclass
class RuntimeContext:
    db: SQLDatabase
    # LangGraph Studio automatically passes these runtime parameters
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    assistant_id: Optional[str] = None
