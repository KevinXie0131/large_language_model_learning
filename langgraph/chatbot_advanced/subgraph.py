"""
LangGraph Subgraph / Graph Nesting Demo
LangGraph 子图/图嵌套演示

Shows how to compose graphs by nesting one graph inside another.
A parent graph delegates part of its work to a child subgraph,
enabling modular, reusable agent components.
展示如何通过将一个图嵌套在另一个图中来组合图。
父图将部分工作委托给子图，实现模块化、可复用的代理组件。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - Subgraph as a node: compiling a child graph and adding it as a node
    子图作为节点：编译子图并将其作为节点添加到父图
  - State mapping: parent and child graphs can have different state schemas
    状态映射：父图和子图可以有不同的状态模式
  - Modularity: subgraphs are self-contained and reusable
    模块化：子图是自包含和可复用的

Graph structure:
图结构：
  Parent: START → router → [research_subgraph] → synthesizer → END
                         → [writing_subgraph]  → synthesizer → END

  Research subgraph: START → search → summarize → END
  Writing subgraph:  START → outline → draft → END
"""

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 1. Define State Schemas
#    Parent and child graphs each have their own state.
# 1. 定义状态模式
#    父图和子图各有自己的状态。
# ---------------------------------------------------------------------------


class ResearchState(TypedDict):
    """State for the research subgraph. / 研究子图的状态。"""
    query: str          # research query / 研究查询
    raw_results: str    # raw search results / 原始搜索结果
    summary: str        # summarized results / 总结后的结果


class WritingState(TypedDict):
    """State for the writing subgraph. / 写作子图的状态。"""
    topic: str          # writing topic / 写作主题
    outline: str        # generated outline / 生成的大纲
    draft: str          # final draft / 最终草稿


class ParentState(TypedDict):
    """State for the parent graph. / 父图的状态。"""
    user_request: str   # the user's original request / 用户的原始请求
    task_type: str      # "research" or "writing" / 任务类型
    result: str         # final result from subgraph / 子图的最终结果


# ---------------------------------------------------------------------------
# 2. Build the Research Subgraph
#    search → summarize
# 2. 构建研究子图
#    搜索 → 总结
# ---------------------------------------------------------------------------


def search_node(state: ResearchState) -> dict:
    """Simulate searching for information on the query.
    模拟根据查询搜索信息。"""
    response = llm.invoke([
        SystemMessage(content="You are a research assistant. Given a query, provide 3-5 key facts or findings. Keep it concise."),
        HumanMessage(content=f"Research this topic: {state['query']}")
    ])
    return {"raw_results": response.content}


def summarize_node(state: ResearchState) -> dict:
    """Summarize the raw research results into a clean summary.
    将原始研究结果总结为简洁的摘要。"""
    response = llm.invoke([
        SystemMessage(content="Summarize the following research findings into a clear, concise paragraph."),
        HumanMessage(content=state["raw_results"])
    ])
    return {"summary": response.content}


# Build and compile the research subgraph / 构建并编译研究子图
research_graph = StateGraph(ResearchState)
research_graph.add_node("search", search_node)
research_graph.add_node("summarize", summarize_node)
research_graph.add_edge(START, "search")         # Entry: start with search / 入口：从搜索开始
research_graph.add_edge("search", "summarize")    # search → summarize / 搜索 → 总结
research_graph.add_edge("summarize", END)          # summarize → end / 总结 → 结束
research_subgraph = research_graph.compile()       # Compile into a runnable / 编译为可执行对象


# ---------------------------------------------------------------------------
# 3. Build the Writing Subgraph
#    outline → draft
# 3. 构建写作子图
#    大纲 → 草稿
# ---------------------------------------------------------------------------


def outline_node(state: WritingState) -> dict:
    """Generate an outline for the writing topic.
    为写作主题生成大纲。"""
    response = llm.invoke([
        SystemMessage(content="Create a brief 3-point outline for a short article on the given topic."),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    return {"outline": response.content}


def draft_node(state: WritingState) -> dict:
    """Write a draft based on the outline.
    根据大纲撰写草稿。"""
    response = llm.invoke([
        SystemMessage(content="Write a short article (2-3 paragraphs) based on this outline."),
        HumanMessage(content=f"Topic: {state['topic']}\n\nOutline:\n{state['outline']}")
    ])
    return {"draft": response.content}


# Build and compile the writing subgraph / 构建并编译写作子图
writing_graph = StateGraph(WritingState)
writing_graph.add_node("outline", outline_node)
writing_graph.add_node("draft", draft_node)
writing_graph.add_edge(START, "outline")          # Entry: start with outline / 入口：从大纲开始
writing_graph.add_edge("outline", "draft")         # outline → draft / 大纲 → 草稿
writing_graph.add_edge("draft", END)               # draft → end / 草稿 → 结束
writing_subgraph = writing_graph.compile()         # Compile into a runnable / 编译为可执行对象


# ---------------------------------------------------------------------------
# 4. Build the Parent Graph
#    Router decides which subgraph to use, then synthesizer formats output.
# 4. 构建父图
#    路由器决定使用哪个子图，然后合成器格式化输出。
# ---------------------------------------------------------------------------


class TaskRouter(BaseModel):
    """Router decision schema. / 路由决策模式。"""
    task_type: Literal["research", "writing"]


def router_node(state: ParentState) -> dict:
    """Determine if the request is a research or writing task.
    判断请求是研究任务还是写作任务。"""
    router_llm = llm.with_structured_output(TaskRouter)
    result = router_llm.invoke([
        SystemMessage(content=(
            "Classify the user request as either 'research' (finding/learning about something) "
            "or 'writing' (creating/composing content)."
        )),
        HumanMessage(content=state["user_request"])
    ])
    return {"task_type": result.task_type}


def research_wrapper(state: ParentState) -> dict:
    """Invoke the research subgraph and map results back to parent state.
    调用研究子图并将结果映射回父图状态。

    This is the key pattern: we invoke the compiled subgraph as a regular
    function, passing in the child's state and reading back its output.
    这是关键模式：我们像普通函数一样调用编译后的子图，传入子图状态并读取其输出。
    """
    child_result = research_subgraph.invoke({"query": state["user_request"]})
    return {"result": child_result["summary"]}  # Map child output → parent state / 子图输出 → 父图状态


def writing_wrapper(state: ParentState) -> dict:
    """Invoke the writing subgraph and map results back to parent state.
    调用写作子图并将结果映射回父图状态。"""
    child_result = writing_subgraph.invoke({"topic": state["user_request"]})
    return {"result": child_result["draft"]}


def synthesizer_node(state: ParentState) -> dict:
    """Format the final output with a header.
    格式化最终输出，添加标题。"""
    header = "📚 Research Result" if state["task_type"] == "research" else "✍️ Writing Result"
    return {"result": f"[{header}]\n\n{state['result']}"}


def route_by_task(state: ParentState) -> str:
    """Conditional edge: route to the appropriate subgraph wrapper.
    条件边：路由到对应的子图包装器。"""
    return state["task_type"]


# Build and compile the parent graph / 构建并编译父图
parent_graph = StateGraph(ParentState)
parent_graph.add_node("router", router_node)
parent_graph.add_node("research", research_wrapper)   # Subgraph as a node / 子图作为节点
parent_graph.add_node("writing", writing_wrapper)      # Subgraph as a node / 子图作为节点
parent_graph.add_node("synthesizer", synthesizer_node)

parent_graph.add_edge(START, "router")
parent_graph.add_conditional_edges("router", route_by_task, {
    "research": "research",  # research tasks → research subgraph / 研究任务 → 研究子图
    "writing": "writing",    # writing tasks → writing subgraph / 写作任务 → 写作子图
})
parent_graph.add_edge("research", "synthesizer")
parent_graph.add_edge("writing", "synthesizer")
parent_graph.add_edge("synthesizer", END)

app = parent_graph.compile()

# ---------------------------------------------------------------------------
# 5. Interactive CLI Loop
# 5. 交互式命令行循环
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Subgraph Demo")
    print("  LangGraph 子图嵌套演示")
    print("  Ask research questions or request writing tasks")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = app.invoke({"user_request": user_input})
        print(f"\n{result['result']}")


if __name__ == "__main__":
    main()
