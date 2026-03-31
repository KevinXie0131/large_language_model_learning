"""
LangGraph Reflection / Self-Correction Demo

Shows how to build an agent that generates content, critiques its own
output, and iteratively refines it until quality is satisfactory.

Key LangGraph concepts demonstrated:
  - Custom state with fields beyond messages (draft, critique, iteration)
  - Iterative loops: generate → reflect → (good enough?) → generate → ...
  - Conditional edges based on custom logic (iteration count, quality)
  - Using different system prompts for different nodes (writer vs. critic)

Graph structure:
  START → writer → critic → (needs revision?) → writer → critic → ... → END
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Custom State
#    Beyond just messages, we track the draft, critique, and iteration count.
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 3  # Maximum reflection iterations / 最大反思迭代次数


class ReflectionState(TypedDict):
    topic: str  # the user's original request / 用户的原始请求主题
    draft: str  # current draft from the writer / 写作者的当前草稿
    critique: str  # feedback from the critic / 评论者的反馈
    iteration: int  # current iteration count / 当前迭代次数
    history: list[str]  # track each draft for comparison / 记录每次草稿，用于对比


# ---------------------------------------------------------------------------
# 2. LLM setup
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # temperature 0.7: moderate creativity, good for writing / 温度0.7：适度创造性，适合写作任务

# ---------------------------------------------------------------------------
# 3. Graph Nodes
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """You are an expert writer. Write or revise content based on
the given topic. If critique/feedback is provided, address each point
specifically to improve the draft. Be concise but thorough.

Output ONLY the revised content, no meta-commentary."""

CRITIC_SYSTEM = """You are a writing critic. Review the draft and provide
specific, actionable feedback. Focus on:
1. Clarity and structure
2. Accuracy and completeness
3. Engagement and readability
4. Any missing important points

Rate the overall quality: EXCELLENT, GOOD, or NEEDS_WORK.
Start your response with the rating on its own line, then provide feedback."""


def writer_node(state: ReflectionState):
    """Generate or revise the draft based on the topic and any critique."""
    messages = [SystemMessage(content=WRITER_SYSTEM)]

    if state.get("critique"):  # Has critique feedback → enter revision mode / 有评论反馈 → 进入修订模式
        messages.append(
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n\n"
                    f"Previous draft:\n{state['draft']}\n\n"
                    f"Critique to address:\n{state['critique']}\n\n"
                    "Please revise the draft to address the critique."
                )
            )
        )
    else:  # First pass → just the topic / 首次撰写 → 只提供主题
        messages.append(
            HumanMessage(content=f"Write about: {state['topic']}")
        )

    response = llm.invoke(messages)
    new_draft = response.content

    history = state.get("history", [])
    history.append(new_draft)

    return {
        "draft": new_draft,
        "iteration": state.get("iteration", 0) + 1,
        "history": history,
    }


def critic_node(state: ReflectionState):
    """Critique the current draft."""
    messages = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Draft (iteration {state['iteration']}):\n{state['draft']}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return {"critique": response.content}


# ---------------------------------------------------------------------------
# 4. Routing: should we revise again or finish?
# ---------------------------------------------------------------------------


def should_revise(state: ReflectionState) -> str:
    """Check if the draft needs more revision."""
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)

    # Stop if we've hit the max iterations / 达到最大迭代次数，强制停止
    if iteration >= MAX_ITERATIONS:
        return "done"

    # Stop if the critic rated it EXCELLENT or GOOD / 质量达标，提前停止
    first_line = critique.strip().split("\n")[0].upper()  # Parse the rating from first line / 解析评论第一行的评级
    if "EXCELLENT" in first_line or "GOOD" in first_line:
        return "done"

    return "revise"  # Needs more revision / 需要继续修订


# ---------------------------------------------------------------------------
# 5. Build the Graph
# ---------------------------------------------------------------------------

graph = StateGraph(ReflectionState)

graph.add_node("writer", writer_node)
graph.add_node("critic", critic_node)

graph.add_edge(START, "writer")  # Entry: start by writing a draft / 入口：先写草稿
graph.add_edge("writer", "critic")  # After writing, pass to critic / 写完后交给评论者
graph.add_conditional_edges(  # After critique, decide whether to revise / 评论后决定是否继续修订
    "critic",
    should_revise,
    {"revise": "writer", "done": END},  # revise → back to writer; done → end / revise → 回到写作者; done → 结束
)

app = graph.compile()

# ---------------------------------------------------------------------------
# 6. Interactive CLI
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Reflection / Self-Correction Demo")
    print("  The agent writes, critiques, and refines its output")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print("\nTry: 'Explain how DNS works in simple terms'")
    print("Or:  'Write a short guide to Python decorators'\n")

    while True:
        topic = input("Topic: ").strip()
        if not topic:
            continue
        if topic.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\n[Starting reflection loop...]\n")

        # Stream updates to see each iteration
        for chunk in app.stream(
            {"topic": topic, "iteration": 0, "history": []},
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                if node_name == "writer":
                    iteration = update.get("iteration", "?")
                    print(f"--- DRAFT (iteration {iteration}) ---")
                    print(update.get("draft", ""))
                    print()
                elif node_name == "critic":
                    print(f"--- CRITIQUE ---")
                    print(update.get("critique", ""))
                    print()

        print("=" * 40)
        print("[Reflection complete]")
        print("=" * 40)
        print()


if __name__ == "__main__":
    main()
