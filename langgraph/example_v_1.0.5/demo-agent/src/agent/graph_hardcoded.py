"""LangGraph agent graph with tool calling.

An agent that uses Tavily search and datetime tools via an OpenAI-compatible LLM.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Annotated, Any, Dict

# from dotenv import load_dotenv  # 加载 .env 环境变量
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage  # 消息类型
from langchain_core.tools import tool  # 工具装饰器
from langchain_tavily import TavilySearch  # Tavily 网络搜索工具
from langchain.chat_models import init_chat_model  # 初始化聊天模型
from langgraph.graph import END, StateGraph  # 图构建器和结束标记
from langgraph.graph.message import add_messages  # 消息累加器（reducer）
from langgraph.prebuilt import ToolNode  # 预构建的工具执行节点
from langgraph.runtime import Runtime  # 运行时上下文
from typing_extensions import TypedDict

# load_dotenv()
# 直接设置环境变量（硬编码）
os.environ["LANGSMITH_PROJECT"] = "new-agent"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

# 星期名称映射：weekday() 返回 0-6 对应周一到周日
WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"]


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model_name: str  # 模型名称，如 "gpt-4o"
    temperature: float  # 温度参数，控制输出随机性


class State(TypedDict):
    """Input state for the agent graph.

    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    # 对话消息列表，使用 add_messages reducer 自动追加新消息
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
    max_results=5,  # 最多返回5条结果
    search_depth="advanced",  # advanced 提供更深入的搜索结果
    include_answer=True,  # 包含AI生成的摘要答案
    include_raw_content=False,  # 不包含原始网页内容
    topic="news",  # 使用新闻主题获取时事信息
)

# 工具列表：包含搜索工具和时间工具
tools = [tavily_tool, get_current_datetime]
# 创建工具执行节点，负责实际调用工具并返回结果
tool_node = ToolNode(tools)


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and invoke the LLM with tools bound."""
    # Get configuration from runtime context
    # 从运行时上下文获取配置参数
    context = runtime.context or {}
    model_name = context.get("model_name", "gpt-4o")  # 默认使用 gpt-4o
    temperature = context.get("temperature", 0.7)  # 默认温度 0.7

    # Get API keys from environment variables
    # 从环境变量获取 API 密钥和基础 URL
    # api_key = os.getenv("OPENAIP_API_KEY", "")
    # api_base = os.getenv("OPENAIP_BASE", "https://api.openai.com/v1")
    # API 密钥和基础 URL（硬编码）
    api_key = ""
    api_base = "https://api.openai.com/v1"

    # Initialize model using init_chat_model
    # 使用 OpenAI 兼容接口初始化模型
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        temperature=temperature,
        api_key=api_key,
        base_url=api_base,
    )

    # Bind tools to model
    # 将工具绑定到模型，使模型能够调用工具
    model_with_tools = model.bind_tools(tools)

    # Add system message with current date if not already present
    # 如果消息列表中没有系统消息，添加包含当前时间的系统提示
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
        messages = [system_msg] + messages  # 将系统消息插入到消息列表最前面

    # Invoke model asynchronously
    # 异步调用模型生成回复
    response = await model_with_tools.ainvoke(messages)

    # Return updated state (add_messages reducer will merge automatically)
    # 返回新消息，add_messages reducer 会自动合并到状态中
    return {"messages": [response]}


async def call_tools(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Execute tools requested by the model."""
    # 执行模型请求的工具调用，并返回工具结果
    result = await tool_node.ainvoke(state)
    return result


def should_continue(state: State) -> str:
    """Route to tools if the last message has tool calls, otherwise end."""
    messages = state["messages"]
    last_message = messages[-1]  # 获取最后一条消息

    # 如果最后一条消息是 AI 消息且包含工具调用，则路由到工具节点
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # 否则结束对话流程
    return END


# Define the graph
# 定义 LangGraph 状态图：agent（模型调用）<-> tools（工具执行）的循环
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("agent", call_model)  # 添加模型调用节点
    .add_node("tools", call_tools)  # 添加工具执行节点
    .add_edge("__start__", "agent")  # 入口 -> agent
    .add_conditional_edges(  # agent 节点的条件路由
        "agent",
        should_continue,
        {
            "tools": "tools",  # 有工具调用 -> 执行工具
            END: END,  # 无工具调用 -> 结束
        },
    )
    .add_edge("tools", "agent")  # 工具执行完毕 -> 返回 agent 继续对话
    .compile(name="Agent with Tool Calling")
)
