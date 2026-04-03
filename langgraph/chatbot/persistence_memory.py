"""
LangGraph Persistence & Memory Demo
LangGraph 持久化与记忆演示

Shows how LangGraph's checkpointer system persists conversation state
across turns and supports multiple independent conversation threads.
展示 LangGraph 的检查点系统如何跨轮次持久化对话状态，并支持多个独立对话线程。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - MemorySaver: in-memory checkpointer that stores graph state
    MemorySaver：存储图状态的内存检查点保存器
  - thread_id: unique identifier for each conversation thread
    thread_id：每个对话线程的唯一标识符
  - State persistence: the graph remembers all previous messages per thread
    状态持久化：图记住每个线程的所有历史消息
  - Thread switching: multiple independent conversations in parallel
    线程切换：支持多个独立的并行对话
  - get_state(): inspect the current state of any thread
    get_state()：检查任意线程的当前状态

Graph structure (same as chatbot.py):
图结构（与 chatbot.py 相同）：
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
# 1. 工具
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time. / 获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def remember_fact(fact: str) -> str:
    """Store a fact that the user wants to remember. The fact is stored
    in the conversation history and will be available in future turns.
    存储用户想记住的事实。该事实保存在对话历史中，在后续对话中可用。"""
    return f"Noted! I'll remember: {fact}"  # Fact stored in conversation history, persisted via checkpointer / 事实存储在对话历史中，通过检查点持久化


tools = [get_current_time, remember_fact]

# ---------------------------------------------------------------------------
# 2. LLM + Graph (same structure as chatbot.py)
# 2. LLM + 图（与 chatbot.py 结构相同）
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
# 3. 使用检查点编译
#    检查点在每个图步骤后保存完整状态。每个 thread_id 有独立的状态。
#    MemorySaver 存储在内存中（进程退出后丢失）。
#    生产环境使用 SqliteSaver 或 PostgresSaver 实现持久存储。
# ---------------------------------------------------------------------------

# MemorySaver stores in RAM (lost when process exits). For real persistence, use SqliteSaver or PostgresSaver
# 内存检查点（进程退出后丢失；生产环境用 SqliteSaver 或 PostgresSaver）
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)  # Compile graph with checkpoint persistence / 编译图并启用检查点持久化

# ---------------------------------------------------------------------------
# 4. Interactive CLI with Thread Management
# 4. 带线程管理的交互式命令行
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
    """Display the message history for the current thread. / 显示当前线程的消息历史。"""
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
