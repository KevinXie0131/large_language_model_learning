"""
LangGraph Streaming Demo
LangGraph 流式输出演示

Shows how to stream LLM tokens and graph events in real-time instead
of waiting for the full response. Demonstrates multiple stream modes.
展示如何实时流式输出 LLM token 和图事件，而不是等待完整响应。演示多种流式模式。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - stream_mode="updates": yields state updates from each node as they complete
    stream_mode="updates"：每个节点完成时产出状态更新
  - stream_mode="messages": yields individual LLM tokens as they're generated
    stream_mode="messages"：LLM 生成 token 时逐个产出
  - Real-time display of which graph nodes are executing
    实时显示正在执行的图节点
  - Token-by-token output for a responsive chat experience
    逐 token 输出，提供响应式聊天体验

Graph structure (same as chatbot.py):
图结构（与 chatbot.py 相同）：
  START → chatbot → (has tool calls?) → tools → chatbot → ... → END
"""

# 这里 stream_mode 有三种选项：
#   updates：按节点执行步骤流式返回 graph state 的更新（包括 tool / LLM / 其他节点）
#   messages：流式输出 LLM 的 message（token 级别或 chunk 级别）
#   values：返回每一步执行后的完整 state（非流式 token，而是完整结果快照）
#   custom：允许在节点/工具内部通过 get_stream_writer() 主动写入自定义流式输出
# 关于流式输出的这几种选项，在后面结合 Graph，会体现出更大的作用。

# LangGraph 的 stream_mode 常见有四种：
# updates：
#   按节点执行顺序流式返回 graph state 的增量更新（每个 node 执行完触发一次）。
# messages：
#   用于 LLM token 流式输出，返回 message chunk（token 级别）。
# values：
#   返回每一步执行后的完整 state（非增量、非 token 流）。
# custom：
#   允许在节点或工具内部通过 get_stream_writer() 主动写入自定义流式输出。

# updates → 看流程（每一步干了啥）
# messages → 看模型说话（token）
# values → 看结果（完整状态）
# custom → 自己往流里塞内容

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk  # Streaming message chunk type / 流式输出的消息片段类型
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Tools (same as chatbot.py)
# 1. 工具（与 chatbot.py 相同）
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time. / 获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 3 * 4' / 计算数学表达式。示例：'2 + 3 * 4'"""
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
    """Search the web for information on a given query. / 根据查询搜索网络信息。"""
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
# 2. LLM + 图
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
# 3. 流式输出函数
# ---------------------------------------------------------------------------


def stream_updates(user_input: str, messages: list):
    """Stream mode: 'updates' - shows state updates from each node.
    流式模式：'updates' - 显示每个节点的状态更新。

    This mode yields a dict for each node that runs, containing the
    state changes that node produced. Good for seeing the graph's
    step-by-step execution.
    此模式为每个运行的节点产出一个字典，包含该节点产生的状态变更。适合查看图的逐步执行过程。
    """
    print("\n--- Streaming with mode='updates' ---")
    messages.append({"role": "user", "content": user_input})

    final_messages = messages
    for chunk in app.stream(  # updates mode: yields state updates as each node completes / updates 模式：每个节点完成时产出状态更新
        {"messages": messages}, stream_mode="updates"
    ):
        for node_name, update in chunk.items():  # chunk is {node_name: state_update} dict / chunk 是 {节点名: 状态更新} 字典
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
    流式模式：'messages' - 逐个流式输出 LLM token。

    This mode yields (message_chunk, metadata) tuples as the LLM
    generates each token. This is what you want for a real-time
    typing effect in a chat UI.
    此模式在 LLM 生成每个 token 时产出 (message_chunk, metadata) 元组。适用于聊天界面的实时打字效果。
    """
    print("\n--- Streaming with mode='messages' ---")
    messages.append({"role": "user", "content": user_input})

    print("\nAssistant: ", end="", flush=True)
    final_content = ""

    for chunk, metadata in app.stream(  # messages mode: yields individual tokens, ideal for real-time typing effect / messages 模式：逐 token 产出，适合实时打字效果
        {"messages": messages}, stream_mode="messages"
    ):
        # Only print AI message tokens (not tool calls or tool results)
        # 只打印 AI 文本 token（跳过工具调用）
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            print(chunk.content, end="", flush=True)  # flush=True ensures immediate display / flush=True 确保立即显示
            final_content += chunk.content

    print()  # newline after streaming completes / 流式完成后换行
    return final_content


# ---------------------------------------------------------------------------
# 4. Interactive CLI
# 4. 交互式命令行
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
