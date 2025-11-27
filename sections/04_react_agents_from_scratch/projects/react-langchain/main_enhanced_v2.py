import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

from callbacks import AgentCallbackHandler

from langsmith import traceable

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")

class TextLengthInput(BaseModel):
    text: str = Field(..., description="The text whose number of characters will be counted.")

    @field_validator("text", mode="before")
    @classmethod
    def clean_text(cls, raw_text):
        cleaned_text = str(raw_text)
        cleaned_text = cleaned_text.replace("\n", "").replace("\r", "")
        cleaned_text = cleaned_text.replace('"', "")
        cleaned_text = cleaned_text.replace("'", "")
        return cleaned_text.strip()

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

def find_tool_by_name(available_tools: List, tool_name_to_find: str):
    for available_tool in available_tools:
        if available_tool.name == tool_name_to_find:
            return available_tool
    raise ValueError(f"Tool with name '{tool_name_to_find}' not found.")

@traceable()
def run_agent():
    available_tools = [get_text_length]

    # Create LLM with bound tools
    generative_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=GOOGLE_API_KEY,
        temperature=0.7,
        callbacks=[AgentCallbackHandler()]
    )

    # Bind tools to the LLM
    llm_with_tools = generative_llm.bind_tools(available_tools)

    # Agent loop using native tool calling
    conversation_messages = []
    user_question = "What is the length of the word 'DOG' in characters?"
    conversation_messages.append(HumanMessage(content=user_question))

    max_llm_interactions = 10
    current_iteration = 0

    while current_iteration < max_llm_interactions:
        current_iteration += 1

        # Invoke the LLM
        llm_response = llm_with_tools.invoke(conversation_messages)
        conversation_messages.append(llm_response)

        # Check if the LLM wants to use tools
        if llm_response.tool_calls:
            # Execute each tool call
            for tool_call_dict in llm_response.tool_calls:
                called_tool_name = tool_call_dict["name"]
                tool_input_arguments = tool_call_dict["args"]

                print(f"\nTool Call: {called_tool_name}")
                print(f"Input: {tool_input_arguments}")

                # Find and execute the tool
                matched_tool = find_tool_by_name(available_tools, called_tool_name)
                tool_result = matched_tool.invoke(tool_input_arguments)

                print(f"Output: {tool_result}")

                # Add tool result to conversation messages
                conversation_messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_dict["id"]
                    )
                )
        else:
            # No tool calls, we have the final answer
            print(f"\nFinal Answer: {llm_response.content}")
            break

    if current_iteration >= max_llm_interactions:
        print(f"\nReached maximum iterations ({max_llm_interactions})")

if __name__ == "__main__":
    run_agent()
