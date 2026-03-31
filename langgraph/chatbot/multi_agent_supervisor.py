"""
LangGraph Multi-Agent Supervisor Demo

Shows how to build a supervisor agent that routes tasks to specialized
worker agents (researcher, coder), then synthesizes their outputs.

Key LangGraph concepts demonstrated:
  - Multi-agent coordination via a supervisor node
  - Structured output (with_structured_output) for routing decisions
  - Typed state with custom fields beyond just messages
  - Worker agents as separate graph nodes with specialized system prompts

Graph structure:
  START → supervisor → (route) → researcher → supervisor
                               → coder      → supervisor
                               → FINISH     → END
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel  # For defining structured output data models / 用于定义结构化输出的数据模型

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define the routing schema
#    The supervisor uses structured output to pick the next worker.
# ---------------------------------------------------------------------------

WORKERS = ["researcher", "coder"]


class RouterOutput(BaseModel):
    """The supervisor's routing decision."""

    next: Literal["researcher", "coder", "FINISH"]  # Next routing target / 下一步路由目标
    reason: str  # Routing reason / 路由原因说明


# ---------------------------------------------------------------------------
# 2. LLM setup
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 3. Supervisor Node
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM = """You are a team supervisor managing two workers:
- researcher: Searches for information, gathers facts, does analysis
- coder: Writes code, debugs, explains technical implementations

Given the conversation so far, decide which worker should act next.
If the task is complete and all information has been gathered/written,
respond with FINISH.

Always explain your routing reason briefly."""


def supervisor_node(state: MessagesState):
    """Decide which worker to call next, or finish."""
    structured_llm = llm.with_structured_output(RouterOutput)  # Constrain LLM output to RouterOutput schema / 约束 LLM 输出为 RouterOutput 结构
    response = structured_llm.invoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM)] + state["messages"]
    )
    if response.next == "FINISH":  # Task complete, no more routing / 任务完成，不再路由
        return {"messages": []}
    return {"messages": [], "next": response.next}  # Store routing decision in state's next field / 将路由决定存入状态的 next 字段


# ---------------------------------------------------------------------------
# 4. Worker Nodes
#    Each worker has a specialized system prompt. Their responses are
#    added back to the shared message history tagged with their name.
# ---------------------------------------------------------------------------

RESEARCHER_SYSTEM = """You are a research assistant. Your job is to:
- Gather relevant information about the topic
- Provide facts, statistics, and context
- Cite sources when possible
- Be thorough but concise

Respond with your research findings."""

CODER_SYSTEM = """You are an expert programmer. Your job is to:
- Write clean, well-commented code
- Explain your implementation choices
- Handle edge cases appropriately
- Follow best practices for the language

Respond with your code and explanation."""


def researcher_node(state: MessagesState):
    """Research worker - gathers information."""
    response = llm.invoke(
        [SystemMessage(content=RESEARCHER_SYSTEM)] + state["messages"]  # Use researcher-specific system prompt / 使用研究员专属系统提示
    )
    return {
        "messages": [
            HumanMessage(content=response.content, name="researcher")  # Tag message source as researcher / 标记消息来源为 researcher
        ]
    }


def coder_node(state: MessagesState):
    """Coder worker - writes and explains code."""
    response = llm.invoke(
        [SystemMessage(content=CODER_SYSTEM)] + state["messages"]  # Use coder-specific system prompt / 使用编码员专属系统提示
    )
    return {
        "messages": [HumanMessage(content=response.content, name="coder")]  # Tag message source as coder / 标记消息来源为 coder
    }


# ---------------------------------------------------------------------------
# 5. Routing Function
# ---------------------------------------------------------------------------


def route_supervisor(state: MessagesState) -> str:
    """Route based on the supervisor's decision stored in state."""
    next_worker = state.get("next", "FINISH")  # Read routing decision from state / 从状态中读取路由决定
    if next_worker == "FINISH":
        return END  # Task complete, end graph / 任务完成，结束图
    return next_worker  # Route to the corresponding worker node / 路由到对应的工作节点


# ---------------------------------------------------------------------------
# 6. Build the Graph
# ---------------------------------------------------------------------------

# Use a custom state that includes a "next" field for routing
# 自定义状态：在 MessagesState 基础上增加 next 字段用于路由
from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages annotation: new messages are appended, not replaced / add_messages 注解：新消息会追加而非替换
    next: str  # Supervisor's routing decision (researcher / coder / FINISH) / 主管的路由决定（researcher / coder / FINISH）


graph = StateGraph(SupervisorState)  # Create graph with custom state / 使用自定义状态创建图

# Add all nodes / 添加所有节点
graph.add_node("supervisor", supervisor_node)  # Supervisor node: decides routing / 主管节点：决定路由
graph.add_node("researcher", researcher_node)  # Researcher node: gathers info / 研究员节点：收集信息
graph.add_node("coder", coder_node)  # Coder node: writes code / 编码员节点：编写代码

# Entry point / 入口
graph.add_edge(START, "supervisor")  # Start with supervisor analyzing the task / 先由主管分析任务
# Supervisor routes to workers or END / 主管根据决定路由到工人或结束
graph.add_conditional_edges("supervisor", route_supervisor)
# Workers always report back to supervisor / 工人完成后回到主管
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")

app = graph.compile()

# ---------------------------------------------------------------------------
# 7. Interactive CLI
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Multi-Agent Supervisor Demo")
    print("  Workers: researcher, coder")
    print("  The supervisor routes your request to the right worker")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print("\nTry: 'Research Python async patterns and write an example'")
    print("Or:  'What are the pros and cons of microservices?'\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\n[Supervisor is coordinating...]\n")

        # Stream events to see which workers are being called
        # 流式输出，实时查看哪些工作节点被调用
        events = app.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="updates",  # updates mode: output state updates as each node completes / updates 模式：每个节点完成时输出状态更新
        )

        for event in events:
            for node_name, update in event.items():
                if node_name == "supervisor":
                    next_worker = update.get("next", "FINISH")
                    if next_worker != "FINISH":
                        print(f"[Supervisor → routing to: {next_worker}]")
                    else:
                        print("[Supervisor → task complete]")
                elif node_name in ("researcher", "coder"):
                    msgs = update.get("messages", [])
                    for msg in msgs:
                        content = (
                            msg.content
                            if hasattr(msg, "content")
                            else str(msg)
                        )
                        print(f"\n--- {node_name.upper()} ---")
                        print(content)
                        print(f"--- END {node_name.upper()} ---\n")

        print()


if __name__ == "__main__":
    main()
