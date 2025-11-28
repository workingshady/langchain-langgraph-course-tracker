import os

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from ..context.runtime_context import RuntimeContext
from .prompts import SYSTEM_PROMPT
from .tools import execute_sql


def build_agent(api_key: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.0)
    return create_agent(
        tools=[execute_sql],
        system_prompt=SYSTEM_PROMPT,
        context_schema=RuntimeContext,
        model=llm
    )
