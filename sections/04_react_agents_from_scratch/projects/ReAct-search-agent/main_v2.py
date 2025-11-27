import os

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

# Environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")
if not TAVILY_API_KEY:
    raise EnvironmentError("TAVILY_API_KEY not set in environment or .env file")
if not LANGSMITH_API_KEY:
    raise EnvironmentError("LANGSMITH_API_KEY not set in environment or .env file")

# Initialize model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.5,
)

# Initialize tools
tools = [
    TavilySearch(api_key=TAVILY_API_KEY)
]

# Create agent with structured output
# Using ToolStrategy for structured output (works with any model that supports tool calling)
agent = create_agent(
    model=model,
    tools=tools,
    response_format=ToolStrategy(AgentResponse)
)


def main():
    """Main function to run the agent."""
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
        }]
    })

    # Access the structured response
    structured_response = result.get("structured_response")

    if structured_response:
        print("=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(structured_response.answer)
        print("\n" + "=" * 80)
        print("SOURCES:")
        print("=" * 80)
        for idx, source in enumerate(structured_response.sources, 1):
            print(f"{idx}. {source.url}")
    else:
        print("No structured response available")
        print("\nFull result:")
        print(result)



if __name__ == "__main__":
    main()


