"""
LangGraph Dynamic Breakpoints / Conditional Interrupt Demo
LangGraph 动态断点/条件中断演示

Shows how to conditionally interrupt graph execution based on runtime state,
allowing human review only when certain thresholds are met.
展示如何根据运行时状态有条件地中断图执行，仅在满足特定阈值时请求人工审核。

Unlike the basic human_in_the_loop.py which always interrupts,
this demo interrupts ONLY when the action is considered high-risk.
与总是中断的 human_in_the_loop.py 不同，此演示仅在操作被认为是高风险时才中断。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - interrupt() with conditional logic: only pause when needed
    带条件逻辑的 interrupt()：仅在需要时暂停
  - Command(resume=...) to continue after interrupt
    Command(resume=...) 在中断后继续执行
  - MemorySaver for state persistence across interrupts
    MemorySaver 在中断间保持状态持久性
  - Risk assessment as a routing mechanism
    风险评估作为路由机制

Graph structure:
图结构：
  START → assess_risk → execute_action →（高风险？interrupt!）→ END
                                       →（低风险？auto-approve）→ END
"""

from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # In-memory checkpointer / 内存检查点保存器
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt  # interrupt() pauses execution / interrupt() 暂停执行
from pydantic import BaseModel

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 1. Define State and Risk Schema
# 1. 定义状态和风险模式
# ---------------------------------------------------------------------------

RISK_THRESHOLD = 7  # Actions scoring >= 7 require human approval / 评分 >= 7 的操作需要人工批准


class RiskAssessment(BaseModel):
    """LLM output schema for risk scoring. / LLM 风险评分输出模式。"""
    risk_score: int          # 1-10 scale / 1-10 评分
    risk_reason: str         # why this score / 评分原因
    suggested_action: str    # what the agent wants to do / 代理想要执行的操作


class ActionState(TypedDict):
    user_request: str        # what the user asked / 用户的请求
    risk_score: int          # assessed risk level / 评估的风险等级
    risk_reason: str         # explanation / 解释
    suggested_action: str    # proposed action / 建议的操作
    approved: bool           # whether human approved / 是否经人工批准
    result: str              # final outcome / 最终结果


# ---------------------------------------------------------------------------
# 2. Define Nodes
# 2. 定义节点
# ---------------------------------------------------------------------------


def assess_risk_node(state: ActionState) -> dict:
    """Analyze the user request and assign a risk score.
    分析用户请求并分配风险评分。"""
    risk_llm = llm.with_structured_output(RiskAssessment)
    assessment = risk_llm.invoke([
        SystemMessage(content=(
            "You are a risk assessment system. Score the risk of the requested action "
            "from 1 (harmless) to 10 (very dangerous). Consider:\n"
            "- Data deletion/modification → high risk (8-10)\n"
            "- Sending emails/messages → medium-high risk (6-8)\n"
            "- Reading/querying data → low risk (1-3)\n"
            "- File creation → low-medium risk (3-5)\n"
            "Provide a specific suggested_action describing what you'd do."
        )),
        HumanMessage(content=state["user_request"])
    ])
    return {
        "risk_score": assessment.risk_score,
        "risk_reason": assessment.risk_reason,
        "suggested_action": assessment.suggested_action,
    }


def execute_action_node(state: ActionState) -> dict:
    """Execute the action, but interrupt if risk is high.
    执行操作，但如果风险高则中断。

    This is the key pattern: interrupt() is called CONDITIONALLY
    inside the node based on the risk score.
    这是关键模式：interrupt() 根据风险评分在节点内部有条件地调用。
    """
    if state["risk_score"] >= RISK_THRESHOLD:
        # High risk → interrupt and ask human for approval
        # 高风险 → 中断并请求人工批准
        human_response = interrupt({
            "question": (
                f"⚠️  HIGH RISK ACTION (score: {state['risk_score']}/10)\n"
                f"Reason: {state['risk_reason']}\n"
                f"Proposed action: {state['suggested_action']}\n\n"
                f"Do you approve? (yes/no)"
            )
        })

        if human_response.lower() not in ("yes", "y"):
            return {
                "approved": False,
                "result": f"❌ Action REJECTED by human. Reason: {human_response}"
            }

    # Low risk or human approved → execute
    # 低风险或人工已批准 → 执行
    response = llm.invoke([
        SystemMessage(content="Simulate executing this action. Describe what you did as if you performed it."),
        HumanMessage(content=f"Action: {state['suggested_action']}")
    ])
    return {
        "approved": True,
        "result": f"✅ Action executed: {response.content}"
    }


# ---------------------------------------------------------------------------
# 3. Build the Graph
# 3. 构建图
# ---------------------------------------------------------------------------

graph = StateGraph(ActionState)
graph.add_node("assess_risk", assess_risk_node)
graph.add_node("execute_action", execute_action_node)

graph.add_edge(START, "assess_risk")
graph.add_edge("assess_risk", "execute_action")
graph.add_edge("execute_action", END)

# MemorySaver is required for interrupt() — state must persist across pauses
# MemorySaver 是 interrupt() 所必需的 — 状态必须在暂停间持久化
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# 4. Interactive CLI Loop
#    Handles both normal execution and interrupt/resume flow.
# 4. 交互式命令行循环
#    处理正常执行和中断/恢复流程。
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Dynamic Breakpoints Demo")
    print("  LangGraph 动态断点演示")
    print(f"  Actions with risk score >= {RISK_THRESHOLD} require approval")
    print("  Type 'quit' to exit")
    print("=" * 60)

    thread_id = 0

    while True:
        user_input = input("\nRequest: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        thread_id += 1
        config = {"configurable": {"thread_id": str(thread_id)}}

        # First invocation — may complete or interrupt
        # 第一次调用 — 可能完成也可能中断
        result = app.invoke(
            {"user_request": user_input, "approved": False},
            config=config
        )

        # Check if we hit an interrupt (state has no result yet)
        # 检查是否遇到中断（状态还没有结果）
        snapshot = app.get_state(config)
        if snapshot.next:  # There are pending nodes → we're interrupted / 有待执行节点 → 已中断
            # The interrupt value contains the question for the human
            # 中断值包含给人类的问题
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    print(f"\n{task.interrupts[0].value['question']}")

            # Get human response / 获取人类响应
            human_input = input("\nYour decision: ").strip()

            # Resume with human's response / 用人类的响应恢复执行
            result = app.invoke(
                Command(resume=human_input),
                config=config
            )

        # Print final result / 打印最终结果
        print(f"\nRisk Score: {result['risk_score']}/10")
        print(f"Risk Reason: {result['risk_reason']}")
        print(f"Result: {result['result']}")


if __name__ == "__main__":
    main()
