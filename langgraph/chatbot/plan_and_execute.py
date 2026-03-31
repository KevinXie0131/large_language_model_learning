"""
LangGraph Plan-and-Execute Demo

Shows how to build an agent that first creates a step-by-step plan,
then executes each step, and re-plans if needed.

Key LangGraph concepts demonstrated:
  - Two-phase agent: planning then execution
  - Custom state with plan list, step tracking, and results
  - Conditional looping: execute steps until plan is complete
  - Dynamic re-planning based on execution results

Graph structure:
  START → planner → executor → (more steps?) → executor → ...
                                  ↘ (done)
                        re-planner → (needs changes?) → executor
                                       ↘ (no)
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
# ---------------------------------------------------------------------------


class PlanExecuteState(TypedDict):
    task: str  # the original user request
    plan: list[str]  # list of steps to execute
    current_step: int  # index of the current step
    results: list[str]  # results from each executed step
    final_answer: str  # the synthesized final answer


# ---------------------------------------------------------------------------
# 2. LLM setup
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 3. Graph Nodes
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
    """Create the initial plan."""
    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Create a plan for: {state['task']}"),
    ]
    response = llm.invoke(messages)

    # Parse the numbered list into steps
    steps = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove the number prefix (e.g., "1. " or "1) ")
            step = line.split(".", 1)[-1].strip() if "." in line[:3] else line
            steps.append(step)

    return {"plan": steps, "current_step": 0, "results": []}


EXECUTOR_SYSTEM = """You are an execution agent. You are given a specific step
to complete as part of a larger plan. Execute the step thoroughly and return
your findings or output.

If you need information from previous steps, it will be provided as context."""


def executor_node(state: PlanExecuteState):
    """Execute the current step of the plan."""
    step_idx = state["current_step"]
    current_step = state["plan"][step_idx]
    previous_results = state.get("results", [])

    # Build context from previous steps
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

    new_results = list(previous_results) + [response.content]
    return {"results": new_results, "current_step": step_idx + 1}


REPLANNER_SYSTEM = """You are a re-planning agent. Given the original task,
the current plan, and results so far, decide if the plan needs adjustment.

If the plan is on track and all steps are done, output: COMPLETE
If the plan needs changes, output a revised plan as a numbered list."""


def replanner_node(state: PlanExecuteState):
    """Check if the plan needs adjustment after executing steps."""
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

    if "COMPLETE" in response.content.upper():
        # Synthesize final answer from all results
        final = "\n\n".join(
            f"**Step {i+1}:** {r}" for i, r in enumerate(state["results"])
        )
        return {"final_answer": final}

    # Parse revised plan
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
# ---------------------------------------------------------------------------


def should_continue_executing(state: PlanExecuteState) -> str:
    """After executing a step, check if there are more steps."""
    if state["current_step"] >= len(state["plan"]):
        return "replanner"
    return "executor"


def after_replan(state: PlanExecuteState) -> str:
    """After re-planning, check if we're done or need to keep going."""
    if state.get("final_answer"):
        return END
    return "executor"


# ---------------------------------------------------------------------------
# 5. Build the Graph
# ---------------------------------------------------------------------------

graph = StateGraph(PlanExecuteState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("replanner", replanner_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue_executing)
graph.add_conditional_edges("replanner", after_replan)

app = graph.compile()

# ---------------------------------------------------------------------------
# 6. Interactive CLI
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
