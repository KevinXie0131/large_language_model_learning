"""
LangGraph Chatbot with Tools - Using Prebuilt create_react_agent
LangGraph 工具聊天机器人 - 使用预构建的 create_react_agent

This is the simplified version of chatbot.py. It uses LangGraph's prebuilt
create_react_agent() helper, which internally builds the same StateGraph
(chatbot node → conditional edge → tool node → loop) in a single call.
这是 chatbot.py 的简化版本。使用 LangGraph 预构建的 create_react_agent() 辅助函数，
它在内部自动构建相同的 StateGraph（chatbot 节点 → 条件边 → tool 节点 → 循环）。

Compare this file with chatbot.py to see the difference between
building a graph from scratch vs. using the prebuilt helper.
将此文件与 chatbot.py 对比，了解从零构建图与使用预构建辅助函数的区别。
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent  # Prebuilt ReAct agent / 预构建的 ReAct 代理

load_dotenv()

# --- Tools (same as chatbot.py) / 工具定义（与 chatbot.py 相同） ---


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


# --- Create the agent in one line (internally builds StateGraph + ToolNode + conditional edges) ---
# --- 一行代码创建代理（内部自动构建 StateGraph + ToolNode + 条件边） ---

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(llm, tools=[get_current_time, calculator, search_web])  # Equivalent to the full graph in chatbot.py / 等价于 chatbot.py 中手动构建的完整图

# --- Interactive CLI ---
# --- 交互式命令行 ---


def main():
    print("=" * 60)
    print("  LangGraph Chatbot (create_react_agent Demo)")
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

        messages.append(HumanMessage(content=user_input))  # Use HumanMessage object instead of dict / 使用 HumanMessage 对象而非字典
        result = agent.invoke({"messages": messages})  # Invoke the prebuilt agent / 调用预构建代理
        ai_message = result["messages"][-1]  # Get the final AI response / 获取最终 AI 回复
        print(f"\nAssistant: {ai_message.content}")
        messages = result["messages"]  # Update conversation history / 更新对话历史


if __name__ == "__main__":
    main()
