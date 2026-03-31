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
from langgraph.prebuilt import ToolNode  # Prebuilt tool execution node / 预构建的工具执行节点

load_dotenv()  # Load .env environment variables / 加载 .env 环境变量

# ---------------------------------------------------------------------------
# 1. Define Tools
#    Tools are plain Python functions decorated with @tool.
#    LangGraph converts them into a schema the LLM can call.
# ---------------------------------------------------------------------------


@tool  # @tool decorator converts function into an LLM-callable tool / @tool 装饰器将函数转换为 LLM 可调用的工具
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Return formatted current time / 返回格式化的当前时间


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 3 * 4'"""
    # Only allow safe math operations / 白名单：只允许安全的数学字符
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains invalid characters."
    try:
        result = eval(expression)  # safe because we filtered characters / 经过字符过滤后安全执行
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def search_web(query: str) -> str:
    """Search the web for information on a given query."""
    # Try to use Tavily if the API key is set / 检查是否配置了 Tavily API 密钥
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch

            tavily = TavilySearch(max_results=3)  # Return up to 3 results / 最多返回3条搜索结果
            results = tavily.invoke(query)
            return str(results)
        except Exception as e:
            return f"Tavily search failed: {e}"

    return (  # No API key - return mock result / 没有 API 密钥时返回模拟结果
        f"[Mock search result] No TAVILY_API_KEY set. "
        f"In production, this would search for: '{query}'"
    )


# Collect all tools into a list / 收集所有工具到列表
tools = [get_current_time, calculator, search_web]

# ---------------------------------------------------------------------------
# 2. Create the LLM and bind tools
#    bind_tools() tells the model what tools are available so it can
#    generate structured tool_calls in its responses.
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Initialize LLM, temperature=0 for deterministic output / 初始化 LLM，温度0表示确定性输出
llm_with_tools = llm.bind_tools(tools)  # Bind tools so the LLM knows what it can call / 绑定工具，让 LLM 知道可以调用哪些工具

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
    response = llm_with_tools.invoke(state["messages"])  # Call LLM with full message history / 用完整对话历史调用 LLM
    return {"messages": [response]}  # Return state update, message will be appended to history / 返回状态更新，消息会被追加到历史中


# ToolNode automatically executes any tool_calls from the last AI message
# ToolNode 自动执行 AI 消息中的 tool_calls
tool_node = ToolNode(tools)

# ---------------------------------------------------------------------------
# 4. Define the Routing Function
#    After the chatbot responds, we check: did it request a tool call?
#    - Yes → route to the "tools" node
#    - No  → route to END (finish the conversation turn)
# ---------------------------------------------------------------------------


def should_use_tools(state: MessagesState) -> str:
    """Conditional edge: check if the last message contains tool calls."""
    last_message = state["messages"][-1]  # Get the last message / 获取最后一条消息
    if last_message.tool_calls:  # If the LLM requested tool calls / 如果 LLM 请求调用工具
        return "tools"  # Route to tools node / 路由到工具节点
    return END  # Otherwise end the conversation turn / 否则结束对话轮次


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
# 创建状态图，使用 MessagesState 作为状态模式
graph = StateGraph(MessagesState)

# Add nodes / 添加节点
graph.add_node("chatbot", chatbot_node)  # Chatbot node: calls LLM / 聊天机器人节点：调用 LLM
graph.add_node("tools", tool_node)  # Tools node: executes tool calls / 工具节点：执行工具调用

# Add edges / 添加边（定义节点间的连接）
graph.add_edge(START, "chatbot")  # Always start with the chatbot / 入口：始终从 chatbot 开始
graph.add_conditional_edges("chatbot", should_use_tools)  # Route based on tool calls / 条件边：根据是否有工具调用来路由
graph.add_edge("tools", "chatbot")  # After tools run, go back to chatbot / 工具执行完后回到 chatbot

# Compile the graph into a runnable / 编译图为可执行对象
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

    messages = []  # Maintain full conversation history / 维护完整的对话历史

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})  # Add user message to history / 添加用户消息到历史

        # Invoke the graph with the full message history / 用完整历史调用图
        result = app.invoke({"messages": messages})

        # The last message in the result is the final AI response
        # 结果中最后一条消息是 AI 的回复
        ai_message = result["messages"][-1]
        print(f"\nAssistant: {ai_message.content}")

        # Update our message history with the full result (includes tool call records)
        # 用图返回的完整消息列表更新历史（包含工具调用记录）
        messages = result["messages"]


if __name__ == "__main__":
    main()
