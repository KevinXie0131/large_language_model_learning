"""LangGraph agent graph with tool calling.

An agent that uses Tavily search and datetime tools via an OpenAI-compatible LLM.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Annotated, Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

load_dotenv()

WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"]


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model_name: str
    temperature: float


class State(TypedDict):
    """Input state for the agent graph.

    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    messages: Annotated[list[AnyMessage], add_messages]


@tool
def get_current_datetime() -> str:
    """获取当前日期和时间。

    当需要知道现在的日期、时间、星期几时使用此工具。

    Returns:
        包含当前日期时间信息的字符串
    """
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}, 星期{WEEKDAYS[now.weekday()]}"


# 配置 Tavily 搜索工具以获取最新信息
tavily_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    topic="news",
)

# 工具列表：包含搜索工具和时间工具
tools = [tavily_tool, get_current_datetime]
tool_node = ToolNode(tools)


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and invoke the LLM with tools bound."""
    # Get configuration from runtime context
    context = runtime.context or {}
    model_name = context.get("model_name", "gpt-4o")
    temperature = context.get("temperature", 0.7)

    # Get API keys from environment variables
    api_key = os.getenv("OPENAIP_API_KEY", "")
    api_base = os.getenv("OPENAIP_BASE", "https://api.openai.com/v1")

    # Initialize model using init_chat_model
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        temperature=temperature,
        api_key=api_key,
        base_url=api_base,
    )

    # Bind tools to model
    model_with_tools = model.bind_tools(tools)

    # Add system message with current date if not already present
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        current_time = datetime.now()
        system_msg = SystemMessage(
            content=(
                f"你是一个智能助手。当前时间是："
                f"{current_time.strftime('%Y年%m月%d日 %H:%M:%S')}，"
                f"星期{WEEKDAYS[current_time.weekday()]}\n\n"
                "在回答问题时：\n"
                "1. 如果需要获取最新信息，使用 tavily_search 工具进行搜索\n"
                "2. 如果需要确认当前时间，使用 get_current_datetime 工具\n"
                "3. 在解读搜索结果时，要根据当前日期正确理解相对时间（今天、明天、昨天等）\n"
                "4. 回答要准确、完整，基于搜索到的最新信息"
            )
        )
        messages = [system_msg] + messages

    # Invoke model asynchronously
    response = await model_with_tools.ainvoke(messages)

    # Return updated state (add_messages reducer will merge automatically)
    return {"messages": [response]}


async def call_tools(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Execute tools requested by the model."""
    result = await tool_node.ainvoke(state)
    return result


def should_continue(state: State) -> str:
    """Route to tools if the last message has tool calls, otherwise end."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("agent", call_model)
    .add_node("tools", call_tools)
    .add_edge("__start__", "agent")
    .add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    .add_edge("tools", "agent")
    .compile(name="Agent with Tool Calling")
)
