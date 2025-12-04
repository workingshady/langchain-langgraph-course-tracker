import asyncio
import os

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from logger import log_error, log_header, log_success, log_warning

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")

@tool
def triple(num: float) -> float:
    """
    Multiply the input number by 3 and return the result as a float.

    Args:
        num (float): The number to be tripled.

    Returns:
        float: The result of num multiplied by 3.
    """
    return float(num * 3)
# Create tools and bind to LLM
tools = [
    TavilySearchResults(max_results=3),
    triple
]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
).bind_tools(tools)

__all__ = ["llm", "tools"]
