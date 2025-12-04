import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from logger import log_error, log_header, log_success, log_warning
from react import llm, tools

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")


SYSTEM_MESSAGE = SystemMessage(content="""
You are a helpful AI assistant with access to tools and use it to answer the user's question.
""")

def should_continue(state: MessagesState) -> str:
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


log_header("ReAct Agent - Nodes Implementation")

def run_agent_reasoning(state: MessagesState) -> MessagesState:

    response = llm.invoke([{
        "role": "system",
        "content": SYSTEM_MESSAGE.content
    }, *state["messages"]])

    return {"messages": [response]}

tool_node = ToolNode(tools)

workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("agent", run_agent_reasoning)
workflow.set_entry_point("agent")
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "__end__": END
    }
)
workflow.add_edge("tools", "agent")
app = workflow.compile()
log_success("ReAct agent graph compiled!")
app.get_graph().draw_mermaid_png(output_file_path="outputs/react-agent-graph_flow.png")




