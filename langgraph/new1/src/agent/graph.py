"""LangGraph chatbot agent.

A conversational agent powered by an LLM with a configurable system prompt.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


class Context(TypedDict, total=False):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    """

    system_prompt: str
    model: str


async def call_model(
    state: MessagesState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Invoke the LLM with the conversation history."""
    context = runtime.context or {}
    system_prompt = context.get(
        "system_prompt", "You are a helpful assistant."
    )
    model_name = context.get("model", "gpt-4o-mini")

    llm = ChatOpenAI(model=model_name)
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


# Define the graph
graph = (
    StateGraph(MessagesState, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="Chatbot Agent")
)
