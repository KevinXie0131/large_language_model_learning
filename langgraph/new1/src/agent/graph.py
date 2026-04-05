"""LangGraph chatbot agent.
LangGraph 聊天机器人代理。

A conversational agent powered by an LLM with a configurable system prompt.
基于 LLM 的对话代理，支持可配置的系统提示词。
"""

from __future__ import annotations  # 启用延迟类型注解

from typing import Any, Dict  # 类型标注

from langchain_openai import ChatOpenAI  # OpenAI 聊天模型
from langgraph.graph import MessagesState, StateGraph  # LangGraph 状态图相关
from langgraph.runtime import Runtime  # LangGraph 运行时
from typing_extensions import TypedDict  # 类型字典


class Context(TypedDict, total=False):
    """Context parameters for the agent.
    代理的上下文参数，用于配置系统提示词和模型。

    Set these when creating assistants OR when invoking the graph.
    """

    system_prompt: str  # 系统提示词
    model: str  # 模型名称


async def call_model(
    state: MessagesState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Invoke the LLM with the conversation history."""
    # 使用对话历史调用 LLM
    context = runtime.context or {}  # 获取运行时上下文
    system_prompt = context.get(
        "system_prompt", "You are a helpful assistant."  # 默认系统提示词
    )
    model_name = context.get("model", "gpt-4o-mini")  # 默认模型名称

    llm = ChatOpenAI(model=model_name)  # 初始化 LLM
    # 将系统消息和用户对话历史拼接
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = await llm.ainvoke(messages)  # 异步调用 LLM
    return {"messages": [response]}


# Define the graph
# 定义并编译聊天机器人图
graph = (
    StateGraph(MessagesState, context_schema=Context)  # 创建状态图，指定上下文模式
    .add_node(call_model)  # 添加模型调用节点
    .add_edge("__start__", "call_model")  # 添加从起始节点到模型调用的边
    .compile(name="Chatbot Agent")  # 编译图
)
