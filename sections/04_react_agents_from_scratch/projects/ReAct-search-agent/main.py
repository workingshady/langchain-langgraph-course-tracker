import os
from tabnanny import verbose

from dotenv import load_dotenv
from langchain_core import output_parsers

load_dotenv()

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

#from langchain import hub  --> hub module was removed
# from langchain.agents import AgentExecutor
# from langchain.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langsmith import Client

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")
if not TAVILY_API_KEY:
    raise EnvironmentError("TAVILY_API_KEY not set in environment or .env file")
if not LANGSMITH_API_KEY:
    raise EnvironmentError("LANGSMITH_API_KEY not set in environment or .env file")


genai_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GOOGLE_API_KEY,
        temperature=0.5,
)
structured_llm=genai_llm.with_structured_output(AgentResponse)

tools = [
    TavilySearch(api_key=TAVILY_API_KEY)
]
smith_client = Client()
prompt_id = "hwchase17/react"
react_prompt = smith_client.pull_prompt("hwchase17/react", include_model=True)

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=[
        "tool_names",
        "input",
        "agent_scratchpad"
    ],
    # partial_variables={"format_instructions": output_parser.get_format_instructions()}
)


agent = create_react_agent(
    llm=genai_llm,
    tools=tools,
    prompt=react_prompt
)
agent_executor = AgentExecutor(agent,tools=tools,verbose=True)
# chain = agent_executor

extract_output = RunnableLambda(lambda x: x["output"])
# parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = (agent_executor
         | extract_output
        #  | parse_output
         | structured_llm)


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print(result)
    #result = agent.run(query)
    #parsed = output_parser.parse(result["output"])


if __name__ == "__main__":
    main()
