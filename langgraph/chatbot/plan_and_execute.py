"""
LangGraph Plan-and-Execute Demo
LangGraph 计划与执行演示

Shows how to build an agent that first creates a step-by-step plan,
then executes each step, and re-plans if needed.
展示如何构建一个先制定分步计划、然后逐步执行、并在需要时重新规划的代理。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - Two-phase agent: planning then execution
    两阶段代理：先规划后执行
  - Custom state with plan list, step tracking, and results
    自定义状态：包含计划列表、步骤跟踪和执行结果
  - Conditional looping: execute steps until plan is complete
    条件循环：执行步骤直到计划完成
  - Dynamic re-planning based on execution results
    基于执行结果的动态重新规划

Graph structure:
图结构：
  START → planner → executor →（还有步骤？）→ executor → ...
                                  ↘（完成）
                        re-planner →（需要调整？）→ executor
                                       ↘（不需要）
                                        END
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Custom State
# 1. 自定义状态
# ---------------------------------------------------------------------------


class PlanExecuteState(TypedDict):
    task: str  # the original user request / 用户的原始任务请求
    plan: list[str]  # list of steps to execute / 待执行的步骤列表
    current_step: int  # index of the current step / 当前执行到第几步（索引）
    results: list[str]  # results from each executed step / 每步执行的结果
    final_answer: str  # the synthesized final answer / 综合后的最终答案


# ---------------------------------------------------------------------------
# 2. LLM setup
# 2. LLM 配置
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 3. Graph Nodes
# 3. 图节点
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are a planning agent. Given a task, create a clear
step-by-step plan to accomplish it. Each step should be concrete and actionable.

Output the plan as a numbered list, one step per line:
1. First step
2. Second step
3. Third step
...

Keep plans to 3-5 steps. The final step should always be to synthesize
results into a final answer."""


def planner_node(state: PlanExecuteState):
    """Create the initial plan. / 创建初始计划。"""
    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Create a plan for: {state['task']}"),
    ]
    response = llm.invoke(messages)

    # Parse the numbered list into steps / 解析编号列表为步骤
    steps = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            step = line.split(".", 1)[-1].strip() if "." in line[:3] else line  # Remove the number prefix / 去除编号前缀
            steps.append(step)

    return {"plan": steps, "current_step": 0, "results": []}  # Initialize plan state / 初始化计划状态


EXECUTOR_SYSTEM = """You are an execution agent. You are given a specific step
to complete as part of a larger plan. Execute the step thoroughly and return
your findings or output.

If you need information from previous steps, it will be provided as context."""


def executor_node(state: PlanExecuteState):
    """Execute the current step of the plan. / 执行计划的当前步骤。"""
    step_idx = state["current_step"]
    current_step = state["plan"][step_idx]  # Get the current step to execute / 获取当前待执行的步骤
    previous_results = state.get("results", [])

    # Build context from previous steps / 构建前序步骤的执行结果作为上下文
    context = ""
    if previous_results:
        context = "\n\nResults from previous steps:\n"
        for i, result in enumerate(previous_results):
            context += f"Step {i + 1}: {result}\n"

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM),
        HumanMessage(
            content=(
                f"Overall task: {state['task']}\n"
                f"Current step ({step_idx + 1}/{len(state['plan'])}): "
                f"{current_step}{context}"
            )
        ),
    ]
    response = llm.invoke(messages)

    new_results = list(previous_results) + [response.content]  # Append current step result / 追加当前步骤结果
    return {"results": new_results, "current_step": step_idx + 1}  # Advance step index / 步骤索引前进


REPLANNER_SYSTEM = """You are a re-planning agent. Given the original task,
the current plan, and results so far, decide if the plan needs adjustment.

If the plan is on track and all steps are done, output: COMPLETE
If the plan needs changes, output a revised plan as a numbered list."""


def replanner_node(state: PlanExecuteState):
    """Check if the plan needs adjustment after executing steps. / 执行步骤后检查计划是否需要调整。"""
    messages = [
        SystemMessage(content=REPLANNER_SYSTEM),
        HumanMessage(
            content=(
                f"Task: {state['task']}\n\n"
                f"Original plan: {state['plan']}\n\n"
                f"Steps completed: {state['current_step']}/{len(state['plan'])}\n\n"
                f"Results so far:\n"
                + "\n".join(
                    f"Step {i+1}: {r}" for i, r in enumerate(state["results"])
                )
            )
        ),
    ]
    response = llm.invoke(messages)

    if "COMPLETE" in response.content.upper():  # Replanner determined task is complete / 重新规划者判断任务已完成
        # Synthesize final answer from all results / 综合所有步骤结果为最终答案
        final = "\n\n".join(
            f"**Step {i+1}:** {r}" for i, r in enumerate(state["results"])
        )
        return {"final_answer": final}

    # Parse revised plan / 解析修订后的计划
    steps = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            step = line.split(".", 1)[-1].strip() if "." in line[:3] else line
            steps.append(step)

    if steps:
        return {"plan": steps, "current_step": state["current_step"]}

    # Fallback: mark as complete
    final = "\n\n".join(
        f"**Step {i+1}:** {r}" for i, r in enumerate(state["results"])
    )
    return {"final_answer": final}


# ---------------------------------------------------------------------------
# 4. Routing Functions
# 4. 路由函数
# ---------------------------------------------------------------------------


def should_continue_executing(state: PlanExecuteState) -> str:
    """After executing a step, check if there are more steps. / 执行一步后检查是否还有更多步骤。"""
    if state["current_step"] >= len(state["plan"]):  # All steps executed / 所有步骤执行完毕
        return "replanner"  # Hand off to replanner for evaluation / 交给重新规划者评估
    return "executor"  # Continue executing next step / 继续执行下一步


def after_replan(state: PlanExecuteState) -> str:
    """After re-planning, check if we're done or need to keep going. / 重新规划后检查是否完成或需要继续。"""
    if state.get("final_answer"):  # Has final answer → task complete / 有最终答案 → 任务完成
        return END
    return "executor"  # New plan needs continued execution / 新计划需要继续执行


# ---------------------------------------------------------------------------
# 5. Build the Graph
# 5. 构建图
# ---------------------------------------------------------------------------

graph = StateGraph(PlanExecuteState)

graph.add_node("planner", planner_node)  # Planner: creates step-by-step plan / 规划者：制定步骤计划
graph.add_node("executor", executor_node)  # Executor: runs steps one by one / 执行者：逐步执行计划
graph.add_node("replanner", replanner_node)  # Replanner: evaluates if plan needs adjustment / 重新规划者：评估是否需要调整计划

graph.add_edge(START, "planner")  # Entry: create the plan first / 入口：先制定计划
graph.add_edge("planner", "executor")  # After planning, start executing / 计划制定后开始执行
graph.add_conditional_edges("executor", should_continue_executing)  # After execution: continue or hand to replanner / 执行后判断：继续执行或交给重新规划者
graph.add_conditional_edges("replanner", after_replan)  # After replan: done or continue executing / 重新规划后判断：完成或继续执行

app = graph.compile()

# ---------------------------------------------------------------------------
# 6. Interactive CLI
# 6. 交互式命令行
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Plan-and-Execute Demo")
    print("  The agent plans steps, executes them, and re-plans if needed")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print("\nTry: 'Compare Python and Rust for building a web API'")
    print("Or:  'Explain how to set up a CI/CD pipeline for a Node.js app'\n")

    while True:
        task = input("Task: ").strip()
        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\n[Planning...]\n")

        for chunk in app.stream(
            {"task": task},
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                if node_name == "planner":
                    plan = update.get("plan", [])
                    print("Plan created:")
                    for i, step in enumerate(plan, 1):
                        print(f"  {i}. {step}")
                    print()

                elif node_name == "executor":
                    step_idx = update.get("current_step", 0)
                    results = update.get("results", [])
                    if results:
                        latest = results[-1]
                        print(f"--- Step {step_idx} Result ---")
                        print(latest[:500] + ("..." if len(latest) > 500 else ""))
                        print()

                elif node_name == "replanner":
                    if update.get("final_answer"):
                        print("=" * 40)
                        print("FINAL ANSWER:")
                        print("=" * 40)
                        print(update["final_answer"])
                    elif update.get("plan"):
                        print("[Re-planned! New steps:]")
                        for i, step in enumerate(update["plan"], 1):
                            print(f"  {i}. {step}")
                        print()

        print()


if __name__ == "__main__":
    main()
