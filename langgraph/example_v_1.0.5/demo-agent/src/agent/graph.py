"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from flask.cli import load_dotenv
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

load_dotenv()

class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model_name: str
    temperature: float

class State(TypedDict):
    """Input
    添加到对话 Ctrl+L 编辑 Ctrl+I

Defines the initial structure of incoming data.
See: https://langchain.ai.github.io/langgraph/concepts/low_level/#state

messages: Annotated[list[AnyMessage], add_messages]
    """
    pass  # Placeholder for the messages field

@tool
def get_current_datetime() -> str:
    """获取当前日期和时间。

    当需要知道现在的日期、时间、星期几时使用此工具。

    Returns:
    包含当前日期时间信息的字符串
    """
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}, 星期[{'-', '-', '-', '-', '-', '-', '-']}"

# 代码解释：
# - `get_current_datetime()`：获取当前日期和时间。
# - `return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"`：将当前日期和时间以字符串形式返回。


# Initialize tools from environment
# 配置 Tavily 搜索工具以获取最新信息

tavily_tool = TavilySearch(
    max_results=5,
    search_depth="advanced", # advanced 提供更深入的搜索结果
    include_answer=True,
    include_raw_content=False,
    topic="news", # 使用 news 主题获取时事信息
    # days=30, # 注意：新版本使用 time_range 参数
    # time_range="month", # 可选: "day", "week", "month", "year"
)

# 工具列表：包含搜索工具和时间工具
tools = [tavily_tool, get_current_datetime]
tool_node = ToolNode(tools)


# Image 6
async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input
    添加到对话Ctrl+L 编辑Ctrl+I

Can use runtime context to alter behavior.
    """

    # Get configuration from runtime context
    context = runtime.context or {}
    model_name = context.get("model_name", "Qwen/Qwen3-8B")
    temperature = context.get("temperature", 0.7)

    # Get API keys from environment variables
    api_key = os.getenv("OPENAIP_API_KEY", "")
    api_base = os.getenv("OPENAIP_BASE", "https://api.siliconflow.cn/v1")

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
            content=f"""你是一个智能助手。当前时间是：{current_time.strftime('%Y年%m月%d日 %H:%M:%S')}，星期[{'-', '—', '三', '五'}]\"

    在回答问题时：
    1. 如果需要获取最新信息，使用 tavily_search 工具进行搜索
    2. 如果需要确认当前时间，使用 get_current_datetime 工具
    3. 在解读搜索结果时，要根据当前日期正确理解相对时间（今天、明天、昨天等）
    4. 回答要准确、完整，基于搜索到的最新信息"""
        }

        messages = [system_msg] + messages

        # Invoke model asynchronously
        response = await model_with_tools.ainvoke(messages)

        # Return updated state (add_messages reducer will merge automatically)
        return {"messages": [response]}

    # Image 1
    async def call_tools(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
        """Execute tools requested by the model.

        Can use runtime context to alter behavior.
        """

        # Execute tools using ToolNode
        result = await tool_node.ainvoke(state)

    # Image 2
    last_message = messages[-1]

    # If the last message has tool calls, route to tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Otherwise, end
    return "_end_"

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
                "_end_": "_end_",
            },
        )
        .add_edge("tools", "agent")
        .compile(name="Agent with Tool Calling")
    )