"""
LangGraph Chatbot with Tools - Built from Scratch with StateGraph

This demo shows how to build a tool-calling chatbot using LangGraph's core
primitives: StateGraph, nodes, edges, and conditional routing.

Key LangGraph concepts demonstrated:
  - StateGraph: the main graph class that manages agent state
  - MessagesState: built-in state schema that tracks a list of messages
  - Nodes: functions that read state, do work, and return state updates
  - Edges: connections between nodes (static and conditional)
  - ToolNode: prebuilt node that executes tool calls from the LLM
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define Tools
#    Tools are plain Python functions decorated with @tool.
#    LangGraph converts them into a schema the LLM can call.
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 3 * 4'"""
    # Only allow safe math operations
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains invalid characters."
    try:
        result = eval(expression)  # safe because we filtered characters
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def search_web(query: str) -> str:
    """Search the web for information on a given query."""
    # Try to use Tavily if the API key is set
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch

            tavily = TavilySearch(max_results=3)
            results = tavily.invoke(query)
            return str(results)
        except Exception as e:
            return f"Tavily search failed: {e}"

    return (
        f"[Mock search result] No TAVILY_API_KEY set. "
        f"In production, this would search for: '{query}'"
    )


# Collect all tools into a list
tools = [get_current_time, calculator, search_web]

# ---------------------------------------------------------------------------
# 2. Create the LLM and bind tools
#    bind_tools() tells the model what tools are available so it can
#    generate structured tool_calls in its responses.
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# 3. Define Graph Nodes
#    Each node is a function that takes the current state and returns
#    a state update (here, a dict with a "messages" key).
# ---------------------------------------------------------------------------


def chatbot_node(state: MessagesState):
    """Call the LLM with the current message history.

    The LLM may respond with plain text OR with tool_calls.
    Either way, we return the response to be appended to state["messages"].
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ToolNode automatically executes any tool_calls from the last AI message
tool_node = ToolNode(tools)

# ---------------------------------------------------------------------------
# 4. Define the Routing Function
#    After the chatbot responds, we check: did it request a tool call?
#    - Yes → route to the "tools" node
#    - No  → route to END (finish the conversation turn)
# ---------------------------------------------------------------------------


def should_use_tools(state: MessagesState) -> str:
    """Conditional edge: check if the last message contains tool calls."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# 5. Build the Graph
#
#    The graph structure looks like this:
#
#    START → chatbot → (has tool calls?) → tools → chatbot → ... → END
#                         ↘ (no)
#                           END
# ---------------------------------------------------------------------------

# Create the graph with MessagesState as the state schema
graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("chatbot", chatbot_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "chatbot")  # Always start with the chatbot
graph.add_conditional_edges("chatbot", should_use_tools)  # Route based on tool calls
graph.add_edge("tools", "chatbot")  # After tools run, go back to chatbot

# Compile the graph into a runnable
app = graph.compile()

# ---------------------------------------------------------------------------
# 6. Interactive CLI Loop
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Chatbot with Tools (StateGraph Demo)")
    print("  Tools: get_current_time, calculator, search_web")
    print("  Type 'quit' to exit")
    print("=" * 60)

    messages = []

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Invoke the graph with the full message history
        result = app.invoke({"messages": messages})

        # The last message in the result is the final AI response
        ai_message = result["messages"][-1]
        print(f"\nAssistant: {ai_message.content}")

        # Update our message history with the full result
        messages = result["messages"]


if __name__ == "__main__":
    main()
