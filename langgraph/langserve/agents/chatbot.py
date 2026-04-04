"""Chatbot agent using StateGraph with mock tools."""

from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    mock_weather = {
        "tokyo": "☀️ 22°C, sunny with light breeze",
        "london": "🌧️ 14°C, overcast with light rain",
        "new york": "⛅ 18°C, partly cloudy",
        "paris": "🌤️ 20°C, mostly sunny",
        "sydney": "☀️ 26°C, clear skies",
    }
    return mock_weather.get(city.lower(), f"🌡️ 20°C, pleasant weather in {city}")


@tool
def get_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 3 * 4'."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only numbers and +-*/.() are allowed"
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


tools = [get_weather, get_time, calculator]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


def chatbot_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


def should_use_tools(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", should_use_tools)
builder.add_edge("tools", "chatbot")

graph = builder.compile()
