from dotenv import load_dotenv

load_dotenv()


def basic_tool_binding():
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        # Simulated API call
        return f"Weather in {city}: 25°C"

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    )

    llm_with_tools = llm.bind_tools([get_weather])

    response = llm_with_tools.invoke("What's the weather in Dubai?")
    print(response.content)
    for tool_call in getattr(response, "tool_calls", []):
        result = get_weather.run(tool_call['args'])
        print(result)


def agent_with_multiple_models():
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_community.llms import HuggingFaceHub
    from langchain_community.tools import TavilySearchResults
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    tools = [TavilySearchResults(max_results=1)]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Gemini agent (Google's free LLM, within API limits)
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    agent_gemini = create_tool_calling_agent(llm_gemini, tools, prompt)
    executor_gemini = AgentExecutor(agent=agent_gemini, tools=tools)
    result_gemini = executor_gemini.invoke({"input": "Weather in Dubai vs SF in Celsius"})
    print("Gemini result:")
    print(result_gemini)


    llm_hf = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.0, "max_new_tokens": 256},
    )
    agent_hf = create_tool_calling_agent(llm_hf, tools, prompt)
    executor_hf = AgentExecutor(agent=agent_hf, tools=tools)
    result_hf = executor_hf.invoke({"input": "Weather in Dubai vs SF in Celsius"})
    print("HuggingFace Mixtral result:")
    print(result_hf)


if __name__ == "__main__":
    agent_with_multiple_models()
