import json
import os
from typing import List, Union

from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish

from callbacks import AgentCallbackHandler

load_dotenv()

from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description_and_args, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")

class TextLengthInput(BaseModel):
    text: str = Field(..., description="The text whose number of characters will be counted.")

    @field_validator("text", mode="before")
    @classmethod
    def clean_text(cls, v):
        v = str(v)
        v = v.replace("\n", "").replace("\r", "")
        v = v.replace('"', "")
        v = v.replace("'", "")
        return v.strip()

@tool(args_schema=TextLengthInput)
def get_text_length(text: str) -> int:
    """
    Returns the number of characters in the given text.

    Args:
        text (str): The text to measure.

    Returns:
        int: The length of the input text.
    """
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name '{tool_name}' not found.")

def main():
    tools = [get_text_length]

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: You must ONLY output either Action/Action Input OR Thought/Final Answer, never both in the same response. Do NOT generate the Observation - it will be provided to you after you output the Action.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(
        template=template
    ).partial(
        tools=render_text_description_and_args(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    genai_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GOOGLE_API_KEY,
        temperature=0,
        stop=["\nObservation"],
        callbacks=[AgentCallbackHandler()]
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | genai_llm
        | ReActSingleInputOutputParser()
    )

    # Agent loop - Use while not isinstance(agent_step, AgentFinish)
    intermediate_steps = []
    question = "What is the length of the word 'DOG' in characters?"

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": question,
            "agent_scratchpad": intermediate_steps
        }
    )

    while not isinstance(agent_step, AgentFinish):
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            # Parse tool_input if it's a string
            if isinstance(tool_input, str):
                tool_input = json.loads(tool_input.replace("'", '"'))

            # Properly invoke the tool with the input dictionary
            observation = tool_to_use.invoke(tool_input)
            intermediate_steps.append((agent_step, str(observation)))

        agent_step = agent.invoke(
            {
                "input": question,
                "agent_scratchpad": intermediate_steps
            }
        )

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)

if __name__ == "__main__":
    main()
