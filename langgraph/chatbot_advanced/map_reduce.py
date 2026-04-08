"""
LangGraph Map-Reduce Demo
LangGraph Map-Reduce 映射-归约演示

Shows how to fan out work to multiple parallel branches (map),
then aggregate all results into a single output (reduce).
展示如何将工作扇出到多个并行分支（映射），然后将所有结果聚合为单个输出（归约）。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - Send() API: dynamically create parallel branches at runtime
    Send() API：运行时动态创建并行分支
  - Custom reducers with Annotated: merge results from parallel branches
    使用 Annotated 自定义归约器：合并并行分支的结果
  - Fan-out / fan-in pattern: one node spawns many, another collects
    扇出/扇入模式：一个节点产生多个分支，另一个节点收集结果

Graph structure:
图结构：
  START → splitter →（Send）→ analyzer[topic1] ─┐
                            → analyzer[topic2] ─┤→ aggregator → END
                            → analyzer[topic3] ─┘
"""

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import Send  # Send() creates dynamic parallel branches / Send() 创建动态并行分支
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 1. Define State with Custom Reducer
#    The reducer function defines how parallel results are merged.
# 1. 定义带自定义归约器的状态
#    归约器函数定义如何合并并行结果。
# ---------------------------------------------------------------------------


def merge_analyses(existing: list[dict], new: list[dict] | dict) -> list[dict]:
    """Custom reducer: append new analysis results to the existing list.
    自定义归约器：将新的分析结果追加到现有列表中。

    This is called each time a parallel branch returns its result.
    每次并行分支返回结果时，都会调用此函数。
    """
    if isinstance(new, dict):
        return existing + [new]
    return existing + new


class MapReduceState(TypedDict):
    topic: str                                                      # main topic to analyze / 要分析的主题
    aspects: list[str]                                              # aspects to analyze in parallel / 要并行分析的方面
    analyses: Annotated[list[dict], merge_analyses]                 # results merged by reducer / 由归约器合并的结果
    final_report: str                                               # aggregated output / 聚合后的输出


class AnalyzerInput(TypedDict):
    """Input state for each parallel analyzer branch.
    每个并行分析分支的输入状态。"""
    topic: str
    aspect: str
    analyses: Annotated[list[dict], merge_analyses]


# ---------------------------------------------------------------------------
# 2. Define Nodes
# 2. 定义节点
# ---------------------------------------------------------------------------


def splitter_node(state: MapReduceState) -> dict:
    """Break the topic into aspects to analyze in parallel.
    将主题拆分为多个方面，以便并行分析。"""
    response = llm.invoke([
        SystemMessage(content=(
            "Given a topic, list exactly 4 distinct aspects to analyze. "
            "Return ONLY a comma-separated list, nothing else. "
            "Example: 'economic impact, social effects, environmental concerns, technological factors'"
        )),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    aspects = [a.strip() for a in response.content.split(",")]
    return {"aspects": aspects}


def analyzer_node(state: AnalyzerInput) -> dict:
    """Analyze one specific aspect of the topic (runs in parallel).
    分析主题的一个特定方面（并行运行）。

    Each parallel branch gets its own copy of this node with a different aspect.
    每个并行分支获得此节点的独立副本，分析不同的方面。
    """
    response = llm.invoke([
        SystemMessage(content="Provide a concise 2-3 sentence analysis of the given aspect of the topic."),
        HumanMessage(content=f"Topic: {state['topic']}\nAspect to analyze: {state['aspect']}")
    ])
    return {
        "analyses": [{"aspect": state["aspect"], "analysis": response.content}]
    }


def aggregator_node(state: MapReduceState) -> dict:
    """Combine all parallel analysis results into a final report (reduce step).
    将所有并行分析结果合并为最终报告（归约步骤）。"""
    analyses_text = "\n\n".join(
        f"**{a['aspect']}**: {a['analysis']}" for a in state["analyses"]
    )
    response = llm.invoke([
        SystemMessage(content=(
            "You are given multiple analyses of different aspects of a topic. "
            "Synthesize them into a cohesive final report with an introduction and conclusion. "
            "Keep the individual aspect analyses as sections."
        )),
        HumanMessage(content=f"Topic: {state['topic']}\n\nAnalyses:\n{analyses_text}")
    ])
    return {"final_report": response.content}


# ---------------------------------------------------------------------------
# 3. Define the Fan-Out Logic with Send()
#    This is the "map" step: for each aspect, we Send() a new branch.
# 3. 使用 Send() 定义扇出逻辑
#    这是"映射"步骤：为每个方面 Send() 一个新分支。
# ---------------------------------------------------------------------------


def fan_out_to_analyzers(state: MapReduceState) -> list[Send]:
    """Create a parallel branch for each aspect using Send().
    使用 Send() 为每个方面创建一个并行分支。

    Send(node_name, input_state) tells LangGraph to create a new
    execution of that node with the given input.
    Send(节点名, 输入状态) 告诉 LangGraph 用给定输入创建该节点的新执行。
    """
    return [
        Send("analyzer", {
            "topic": state["topic"],
            "aspect": aspect,
            "analyses": [],
        })
        for aspect in state["aspects"]
    ]


# ---------------------------------------------------------------------------
# 4. Build the Graph
# 4. 构建图
# ---------------------------------------------------------------------------

graph = StateGraph(MapReduceState)

graph.add_node("splitter", splitter_node)       # Break topic into aspects / 将主题拆分为多个方面
graph.add_node("analyzer", analyzer_node)       # Analyze one aspect (runs N times in parallel) / 分析一个方面（并行运行 N 次）
graph.add_node("aggregator", aggregator_node)   # Combine all results / 合并所有结果

graph.add_edge(START, "splitter")
graph.add_conditional_edges("splitter", fan_out_to_analyzers)  # Fan-out: Send() to parallel branches / 扇出：Send() 到并行分支
graph.add_edge("analyzer", "aggregator")                        # Each branch feeds into aggregator / 每个分支的结果流入聚合器
graph.add_edge("aggregator", END)

app = graph.compile()

# ---------------------------------------------------------------------------
# 5. Interactive CLI Loop
# 5. 交互式命令行循环
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Map-Reduce Demo")
    print("  LangGraph 映射-归约演示")
    print("  Enter a topic to analyze from multiple angles")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        user_input = input("\nTopic: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nAnalyzing from multiple angles in parallel...")
        result = app.invoke({"topic": user_input, "analyses": []})

        print(f"\n{'=' * 40}")
        print("FINAL REPORT")
        print(f"{'=' * 40}")
        print(result["final_report"])

        print(f"\n[Analyzed {len(result['analyses'])} aspects: {', '.join(a['aspect'] for a in result['analyses'])}]")


if __name__ == "__main__":
    main()
