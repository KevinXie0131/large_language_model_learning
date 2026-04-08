"""
LangGraph Branching & Merging (Fan-Out / Fan-In) Demo
LangGraph 分支与合并（扇出/扇入）演示

Shows how to run multiple nodes in parallel using static fan-out,
then merge their results into a single downstream node.
展示如何使用静态扇出并行运行多个节点，然后将结果合并到一个下游节点中。

Unlike map_reduce.py which uses dynamic Send() for variable branches,
this demo uses static edges to define fixed parallel paths.
与 map_reduce.py 使用动态 Send() 创建可变分支不同，此演示使用静态边定义固定的并行路径。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - Static fan-out: one node feeding into multiple parallel nodes
    静态扇出：一个节点同时连接到多个并行节点
  - Fan-in: multiple nodes converging into one downstream node
    扇入：多个节点汇聚到一个下游节点
  - Annotated reducers: merging parallel updates to shared state
    Annotated 归约器：合并并行更新到共享状态
  - Parallel execution of independent analysis tasks
    并行执行独立的分析任务

Graph structure:
图结构：
                    ┌→ sentiment_analyzer ──┐
  START → input → ├→ keyword_extractor  ──┤→ merger → END
                    └→ language_detector  ──┘
"""

import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 1. Define State with Reducers for Parallel Merging
#    operator.add merges lists from parallel branches.
# 1. 定义带并行合并归约器的状态
#    operator.add 合并来自并行分支的列表。
# ---------------------------------------------------------------------------


class AnalysisState(TypedDict):
    text: str                                                    # input text / 输入文本
    analyses: Annotated[list[dict], operator.add]                # merged results from all branches / 所有分支的合并结果
    final_report: str                                            # combined analysis report / 综合分析报告


# ---------------------------------------------------------------------------
# 2. Define Parallel Nodes (each runs independently)
# 2. 定义并行节点（每个独立运行）
# ---------------------------------------------------------------------------


def input_node(state: AnalysisState) -> dict:
    """Pass-through node that validates input.
    透传节点，验证输入。"""
    print(f"  📥 Received text ({len(state['text'])} chars)")
    return {}


def sentiment_analyzer(state: AnalysisState) -> dict:
    """Analyze the sentiment of the text (runs in parallel).
    分析文本的情感（并行运行）。"""
    response = llm.invoke([
        SystemMessage(content=(
            "Analyze the sentiment of this text. "
            "Respond with: sentiment (positive/negative/neutral/mixed), "
            "confidence (high/medium/low), and a one-sentence explanation."
        )),
        HumanMessage(content=state["text"])
    ])
    print("  ✅ Sentiment analysis complete")
    return {"analyses": [{"type": "sentiment", "result": response.content}]}


def keyword_extractor(state: AnalysisState) -> dict:
    """Extract key topics and keywords (runs in parallel).
    提取关键主题和关键词（并行运行）。"""
    response = llm.invoke([
        SystemMessage(content=(
            "Extract the top 5 keywords/key phrases from this text. "
            "Return them as a comma-separated list with brief relevance notes."
        )),
        HumanMessage(content=state["text"])
    ])
    print("  ✅ Keyword extraction complete")
    return {"analyses": [{"type": "keywords", "result": response.content}]}


def language_detector(state: AnalysisState) -> dict:
    """Detect the language and writing style (runs in parallel).
    检测语言和写作风格（并行运行）。"""
    response = llm.invoke([
        SystemMessage(content=(
            "Analyze this text for: 1) language, 2) formality level, "
            "3) writing style (academic/casual/technical/narrative), "
            "4) estimated reading level. Be concise."
        )),
        HumanMessage(content=state["text"])
    ])
    print("  ✅ Language detection complete")
    return {"analyses": [{"type": "language", "result": response.content}]}


def merger_node(state: AnalysisState) -> dict:
    """Combine all parallel analysis results into a final report.
    将所有并行分析结果合并为最终报告。

    This node runs only after ALL parallel branches complete.
    LangGraph automatically waits for all fan-out branches before fan-in.
    此节点仅在所有并行分支完成后运行。
    LangGraph 自动等待所有扇出分支完成后再扇入。
    """
    analyses_text = "\n\n".join(
        f"[{a['type'].upper()}]\n{a['result']}" for a in state["analyses"]
    )

    response = llm.invoke([
        SystemMessage(content=(
            "Combine these text analyses into a cohesive report. "
            "Use clear section headers and provide a brief executive summary at the top."
        )),
        HumanMessage(content=f"Original text: {state['text'][:200]}...\n\nAnalyses:\n{analyses_text}")
    ])

    return {"final_report": response.content}


# ---------------------------------------------------------------------------
# 3. Build the Graph with Static Fan-Out / Fan-In
#    The key is that input connects to 3 nodes, and all 3 connect to merger.
# 3. 使用静态扇出/扇入构建图
#    关键是 input 连接到3个节点，而3个节点都连接到 merger。
# ---------------------------------------------------------------------------

graph = StateGraph(AnalysisState)

# Add all nodes / 添加所有节点
graph.add_node("input", input_node)
graph.add_node("sentiment_analyzer", sentiment_analyzer)
graph.add_node("keyword_extractor", keyword_extractor)
graph.add_node("language_detector", language_detector)
graph.add_node("merger", merger_node)

# Entry edge / 入口边
graph.add_edge(START, "input")

# Fan-out: input → three parallel analyzers
# 扇出：input → 三个并行分析器
graph.add_edge("input", "sentiment_analyzer")
graph.add_edge("input", "keyword_extractor")
graph.add_edge("input", "language_detector")

# Fan-in: all three analyzers → merger
# 扇入：三个分析器 → 合并器
graph.add_edge("sentiment_analyzer", "merger")
graph.add_edge("keyword_extractor", "merger")
graph.add_edge("language_detector", "merger")

# Exit / 出口
graph.add_edge("merger", END)

app = graph.compile()

# ---------------------------------------------------------------------------
# 4. Interactive CLI Loop
# 4. 交互式命令行循环
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Branching & Merging Demo")
    print("  LangGraph 分支与合并演示")
    print("  Enter text to analyze from 3 angles in parallel")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        print("\nEnter text to analyze (or paste multiple lines, then enter empty line):")
        lines = []
        while True:
            line = input("> " if not lines else "  ").strip()
            if not line:
                break
            if line.lower() in ("quit", "exit", "q") and not lines:
                print("Goodbye!")
                return
            lines.append(line)

        if not lines:
            continue

        text = " ".join(lines)
        print(f"\n🔄 Analyzing in parallel...")

        result = app.invoke({"text": text, "analyses": []})

        print(f"\n{'=' * 40}")
        print("ANALYSIS REPORT")
        print(f"{'=' * 40}")
        print(result["final_report"])
        print(f"\n[Completed {len(result['analyses'])} parallel analyses]")


if __name__ == "__main__":
    main()
