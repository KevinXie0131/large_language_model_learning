"""
LangGraph Streaming Demo

Shows how to stream LLM tokens and graph events in real-time instead
of waiting for the full response. Demonstrates multiple stream modes.

Key LangGraph concepts demonstrated:
  - stream_mode="updates": yields state updates from each node as they complete
  - stream_mode="messages": yields individual LLM tokens as they're generated
  - Real-time display of which graph nodes are executing
  - Token-by-token output for a responsive chat experience

Graph structure (same as chatbot.py):
  START → chatbot → (has tool calls?) → tools → chatbot → ... → END
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Tools (same as chatbot.py)
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 3 * 4'"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains invalid characters."
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def search_web(query: str) -> str:
    """Search the web for information on a given query."""
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch

            tavily = TavilySearch(max_results=3)
            results = tavily.invoke(query)
            return str(results)
        except Exception as e:
            return f"Tavily search failed: {e}"
    return f"[Mock search] No TAVILY_API_KEY set. Query: '{query}'"


tools = [get_current_time, calculator, search_web]

# ---------------------------------------------------------------------------
# 2. LLM + Graph
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def chatbot_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_use_tools(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(tools)

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", should_use_tools)
graph.add_edge("tools", "chatbot")

app = graph.compile()

# ---------------------------------------------------------------------------
# 3. Streaming Functions
# ---------------------------------------------------------------------------


def stream_updates(user_input: str, messages: list):
    """Stream mode: 'updates' - shows state updates from each node.

    This mode yields a dict for each node that runs, containing the
    state changes that node produced. Good for seeing the graph's
    step-by-step execution.
    """
    print("\n--- Streaming with mode='updates' ---")
    messages.append({"role": "user", "content": user_input})

    final_messages = messages
    for chunk in app.stream(
        {"messages": messages}, stream_mode="updates"
    ):
        for node_name, update in chunk.items():
            print(f"\n[Node: {node_name}]")
            node_messages = update.get("messages", [])
            for msg in node_messages:
                if hasattr(msg, "type"):
                    if msg.type == "ai" and msg.content:
                        print(f"  AI: {msg.content}")
                    elif msg.type == "ai" and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(
                                f"  Tool call: {tc['name']}({tc['args']})"
                            )
                    elif msg.type == "tool":
                        print(f"  Tool result ({msg.name}): {msg.content}")

            if "messages" in update:
                final_messages = update["messages"]

    return final_messages


def stream_tokens(user_input: str, messages: list):
    """Stream mode: 'messages' - streams individual LLM tokens.

    This mode yields (message_chunk, metadata) tuples as the LLM
    generates each token. This is what you want for a real-time
    typing effect in a chat UI.
    """
    print("\n--- Streaming with mode='messages' ---")
    messages.append({"role": "user", "content": user_input})

    print("\nAssistant: ", end="", flush=True)
    final_content = ""

    for chunk, metadata in app.stream(
        {"messages": messages}, stream_mode="messages"
    ):
        # Only print AI message tokens (not tool calls or tool results)
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            print(chunk.content, end="", flush=True)
            final_content += chunk.content

    print()  # newline after streaming completes
    return final_content


# ---------------------------------------------------------------------------
# 4. Interactive CLI
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Streaming Demo")
    print("  Shows real-time token streaming and graph step updates")
    print("  Tools: get_current_time, calculator, search_web")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print("\nStreaming modes:")
    print("  [1] 'updates' - see graph node execution step by step")
    print("  [2] 'messages' - see LLM tokens as they're generated")
    print()

    mode = input("Choose mode (1 or 2, default=2): ").strip()
    use_token_mode = mode != "1"

    if use_token_mode:
        print("\nUsing token streaming mode (messages)")
    else:
        print("\nUsing node update streaming mode (updates)")

    messages = []

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if use_token_mode:
            stream_tokens(user_input, messages)
        else:
            messages = stream_updates(user_input, messages)


if __name__ == "__main__":
    main()
