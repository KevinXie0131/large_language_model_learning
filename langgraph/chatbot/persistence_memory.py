"""
LangGraph Persistence & Memory Demo

Shows how LangGraph's checkpointer system persists conversation state
across turns and supports multiple independent conversation threads.

Key LangGraph concepts demonstrated:
  - MemorySaver: in-memory checkpointer that stores graph state
  - thread_id: unique identifier for each conversation thread
  - State persistence: the graph remembers all previous messages per thread
  - Thread switching: multiple independent conversations in parallel
  - get_state(): inspect the current state of any thread

Graph structure (same as chatbot.py):
  START → chatbot → (has tool calls?) → tools → chatbot → ... → END
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # In-memory checkpointer: persist conversation state / 内存检查点：持久化对话状态
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def remember_fact(fact: str) -> str:
    """Store a fact that the user wants to remember. The fact is stored
    in the conversation history and will be available in future turns."""
    return f"Noted! I'll remember: {fact}"  # Fact stored in conversation history, persisted via checkpointer / 事实存储在对话历史中，通过检查点持久化


tools = [get_current_time, remember_fact]

# ---------------------------------------------------------------------------
# 2. LLM + Graph (same structure as chatbot.py)
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

# ---------------------------------------------------------------------------
# 3. Compile with Checkpointer
#    The checkpointer stores the full state after each graph step.
#    Each thread_id gets its own independent state.
#
#    MemorySaver stores in RAM (lost when process exits).
#    For real persistence, use SqliteSaver or PostgresSaver:
#      from langgraph.checkpoint.sqlite import SqliteSaver
#      checkpointer = SqliteSaver.from_conn_string("chat.db")
# ---------------------------------------------------------------------------

# MemorySaver stores in RAM (lost when process exits). For real persistence, use SqliteSaver or PostgresSaver
# 内存检查点（进程退出后丢失；生产环境用 SqliteSaver 或 PostgresSaver）
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)  # Compile graph with checkpoint persistence / 编译图并启用检查点持久化

# ---------------------------------------------------------------------------
# 4. Interactive CLI with Thread Management
# ---------------------------------------------------------------------------

HELP_TEXT = """
Commands:
  /threads          - List all conversation threads
  /switch <name>    - Switch to a different thread (creates if new)
  /history          - Show current thread's message history
  /new <name>       - Create and switch to a new thread
  quit              - Exit
"""


def show_thread_history(config):
    """Display the message history for the current thread."""
    state = app.get_state(config)  # Read full state of specified thread from checkpointer / 从检查点读取指定线程的完整状态
    if not state.values:
        print("  (no messages yet)")
        return

    messages = state.values.get("messages", [])
    for msg in messages:
        role = msg.type if hasattr(msg, "type") else "unknown"
        content = msg.content if hasattr(msg, "content") else str(msg)
        if role == "human":
            print(f"  You: {content}")
        elif role == "ai" and content:
            print(f"  AI:  {content}")
        elif role == "tool":
            print(f"  Tool({msg.name}): {content}")


def main():
    print("=" * 60)
    print("  LangGraph Persistence & Memory Demo")
    print("  Conversations persist across turns via checkpointing")
    print("  Switch between independent threads with /switch")
    print("  Type /help for commands, 'quit' to exit")
    print("=" * 60)

    current_thread = "default"  # Default thread name / 默认线程名
    threads = {"default"}  # Track all created threads / 跟踪所有已创建的线程

    print(f"\n[Active thread: {current_thread}]")

    while True:
        user_input = input(f"\n[{current_thread}] You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Handle commands
        if user_input == "/help":
            print(HELP_TEXT)
            continue

        if user_input == "/threads":
            print(f"  Threads: {', '.join(sorted(threads))}")
            print(f"  Active:  {current_thread}")
            continue

        if user_input.startswith("/switch "):
            thread_name = user_input.split(" ", 1)[1].strip()
            current_thread = thread_name
            threads.add(thread_name)
            print(f"  [Switched to thread: {current_thread}]")
            continue

        if user_input.startswith("/new "):
            thread_name = user_input.split(" ", 1)[1].strip()
            current_thread = thread_name
            threads.add(thread_name)
            print(f"  [Created and switched to thread: {current_thread}]")
            continue

        if user_input == "/history":
            config = {"configurable": {"thread_id": current_thread}}
            show_thread_history(config)
            continue

        # Normal message - send to the graph / 普通消息 - 发送到图
        config = {"configurable": {"thread_id": current_thread}}  # Each thread has independent conversation state / 每个线程有独立的对话状态

        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,  # Checkpointer auto saves/restores full history for this thread / 检查点自动保存/恢复该线程的完整历史
        )

        ai_message = result["messages"][-1]
        print(f"\nAssistant: {ai_message.content}")


if __name__ == "__main__":
    main()
