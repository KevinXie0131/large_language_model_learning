"""ReAct agent using create_react_agent with mock tools."""

from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    mock_results = {
        "langgraph": (
            "LangGraph is a framework for building stateful, multi-actor "
            "applications with LLMs. It extends LangChain with cyclic graph "
            "support for complex agent workflows."
        ),
        "langserve": (
            "LangServe helps deploy LangChain runnables and chains as REST APIs. "
            "It provides /invoke, /stream, and /playground endpoints automatically."
        ),
        "python": (
            "Python is a high-level programming language known for its simplicity. "
            "Python 3.12 is the latest stable release with improved performance."
        ),
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': No specific results found. This is a mock search tool for demonstration purposes."


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


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
graph = create_react_agent(llm, tools=[search_web, get_time, calculator])
