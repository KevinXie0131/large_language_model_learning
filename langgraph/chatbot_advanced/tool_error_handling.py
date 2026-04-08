"""
LangGraph Tool Error Handling & Retry Demo
LangGraph 工具错误处理与重试演示

Shows how to build an agent that gracefully handles tool failures,
retries with corrections, and falls back when tools keep failing.
展示如何构建一个优雅处理工具故障、自动修正重试、持续失败时回退的代理。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - Custom state tracking: error counts and retry logic
    自定义状态跟踪：错误计数和重试逻辑
  - Conditional routing based on error state
    基于错误状态的条件路由
  - Error recovery patterns: retry → correct → fallback
    错误恢复模式：重试 → 修正 → 回退
  - ToolNode with custom error handling wrapper
    带自定义错误处理包装器的 ToolNode

Graph structure:
图结构：
  START → chatbot → (tool calls?) → try_tools → (success?) → chatbot → END
                                              → (error?)   → error_handler → chatbot
                                              → (max retries?) → fallback → END
"""

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MAX_RETRIES = 2  # Maximum retry attempts per tool call / 每次工具调用的最大重试次数

# ---------------------------------------------------------------------------
# 1. Define Tools (some intentionally unreliable)
# 1. 定义工具（部分故意设计为不可靠的）
# ---------------------------------------------------------------------------


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b. Returns the result."""
    if b == 0:
        raise ValueError("Division by zero is not allowed!")
    return str(a / b)


@tool
def lookup_user(user_id: str) -> str:
    """Look up a user by their ID. Valid IDs are 'U001' to 'U005'."""
    valid_users = {
        "U001": "Alice (Engineering)",
        "U002": "Bob (Marketing)",
        "U003": "Charlie (Sales)",
        "U004": "Diana (HR)",
        "U005": "Eve (Finance)",
    }
    if user_id not in valid_users:
        raise ValueError(f"User '{user_id}' not found. Valid IDs are: {', '.join(valid_users.keys())}")
    return valid_users[user_id]


@tool
def get_weather(city: str) -> str:
    """Get weather for a city. Only works for major cities."""
    weather_data = {
        "new york": "72°F, Sunny",
        "london": "58°F, Cloudy",
        "tokyo": "68°F, Clear",
        "paris": "64°F, Rainy",
    }
    city_lower = city.lower()
    if city_lower not in weather_data:
        raise ValueError(f"Weather data not available for '{city}'. Available cities: {', '.join(weather_data.keys())}")
    return f"Weather in {city}: {weather_data[city_lower]}"


tools = [divide, lookup_user, get_weather]
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# 2. Custom State with Error Tracking
# 2. 带错误跟踪的自定义状态
# ---------------------------------------------------------------------------


def merge_messages(existing: list, new: list) -> list:
    return existing + new


class ErrorTrackingState(TypedDict):
    messages: Annotated[list, merge_messages]
    error_count: int          # consecutive errors / 连续错误次数
    last_error: str           # last error message / 最后的错误消息


# ---------------------------------------------------------------------------
# 3. Define Nodes
# 3. 定义节点
# ---------------------------------------------------------------------------


def chatbot_node(state: ErrorTrackingState) -> dict:
    """Call the LLM. Include error context if retrying.
    调用 LLM。如果在重试则包含错误上下文。"""
    messages = state["messages"]

    # If we had an error, add context so the LLM can self-correct
    # 如果有错误，添加上下文让 LLM 自我修正
    if state.get("error_count", 0) > 0 and state.get("last_error"):
        messages = messages + [
            SystemMessage(content=(
                f"Your previous tool call failed with error: {state['last_error']}. "
                f"Please try again with corrected parameters. "
                f"Attempt {state['error_count']}/{MAX_RETRIES}."
            ))
        ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def try_tools_node(state: ErrorTrackingState) -> dict:
    """Execute tool calls with error catching.
    执行工具调用并捕获错误。

    Instead of using the default ToolNode (which would crash on errors),
    we manually execute tools and catch exceptions.
    我们不使用默认的 ToolNode（遇到错误会崩溃），而是手动执行工具并捕获异常。
    """
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        # Find the matching tool / 找到匹配的工具
        tool_fn = {t.name: t for t in tools}.get(tool_call["name"])

        if tool_fn is None:
            results.append(ToolMessage(
                content=f"Error: Unknown tool '{tool_call['name']}'",
                tool_call_id=tool_call["id"],
            ))
            continue

        try:
            # Attempt to execute the tool / 尝试执行工具
            result = tool_fn.invoke(tool_call["args"])
            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            ))
        except Exception as e:
            # Catch the error and return it as a tool message
            # 捕获错误并作为工具消息返回
            error_msg = f"Tool error: {e}"
            results.append(ToolMessage(
                content=error_msg,
                tool_call_id=tool_call["id"],
            ))
            return {
                "messages": results,
                "error_count": state.get("error_count", 0) + 1,
                "last_error": str(e),
            }

    # All tools succeeded / 所有工具执行成功
    return {
        "messages": results,
        "error_count": 0,
        "last_error": "",
    }


def fallback_node(state: ErrorTrackingState) -> dict:
    """Generate a helpful response when tools keep failing.
    当工具持续失败时生成有用的回复。"""
    response = llm.invoke([
        SystemMessage(content=(
            "Your tools have failed multiple times. Provide a helpful response "
            "explaining what went wrong and suggest alternatives. "
            "Be apologetic and constructive."
        )),
        HumanMessage(content=f"The user asked: {state['messages'][0].content if state['messages'] else 'unknown'}"),
        HumanMessage(content=f"Last error: {state.get('last_error', 'unknown')}"),
    ])
    return {
        "messages": [response],
        "error_count": 0,
    }


# ---------------------------------------------------------------------------
# 4. Routing Functions
# 4. 路由函数
# ---------------------------------------------------------------------------


def should_use_tools(state: ErrorTrackingState) -> str:
    """Route: does the LLM want to call tools?
    路由：LLM 是否想调用工具？"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "try_tools"
    return END


def handle_tool_result(state: ErrorTrackingState) -> str:
    """Route after tool execution: success, retry, or fallback.
    工具执行后路由：成功、重试或回退。"""
    if state.get("error_count", 0) == 0:
        return "chatbot"  # Success → back to chatbot / 成功 → 回到聊天机器人
    if state["error_count"] >= MAX_RETRIES:
        return "fallback"  # Too many failures → fallback / 失败次数过多 → 回退
    return "chatbot"  # Error but retries left → retry / 有错误但还有重试次数 → 重试


# ---------------------------------------------------------------------------
# 5. Build the Graph
# 5. 构建图
# ---------------------------------------------------------------------------

graph = StateGraph(ErrorTrackingState)

graph.add_node("chatbot", chatbot_node)
graph.add_node("try_tools", try_tools_node)
graph.add_node("fallback", fallback_node)

graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", should_use_tools)
graph.add_conditional_edges("try_tools", handle_tool_result, {
    "chatbot": "chatbot",    # Success or retry / 成功或重试
    "fallback": "fallback",  # Max retries exceeded / 超过最大重试次数
})
graph.add_edge("fallback", END)

app = graph.compile()

# ---------------------------------------------------------------------------
# 6. Interactive CLI Loop
# 6. 交互式命令行循环
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Tool Error Handling Demo")
    print("  LangGraph 工具错误处理演示")
    print("  Tools: divide, lookup_user, get_weather")
    print("  Try: 'divide 10 by 0', 'find user U999', 'weather in Mars'")
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

        messages.append(HumanMessage(content=user_input))

        result = app.invoke({"messages": messages, "error_count": 0, "last_error": ""})

        ai_message = result["messages"][-1]
        print(f"\nAssistant: {ai_message.content}")

        if result.get("error_count", 0) > 0:
            print(f"  [⚠️ Encountered {result['error_count']} error(s)]")

        messages = result["messages"]


if __name__ == "__main__":
    main()
