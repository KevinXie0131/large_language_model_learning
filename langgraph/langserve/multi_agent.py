"""多 Agent 监督系统，将任务路由给研究员和程序员 Agent。"""

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, messages_from_dict
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


class SupervisorState(TypedDict):
    """监督系统的状态定义。"""
    messages: Annotated[list[BaseMessage], add_messages]  # 消息列表，使用 add_messages 进行合并
    next: str  # 下一个要执行的节点名称


class RouterOutput(BaseModel):
    """路由器输出模型，用于结构化 LLM 的路由决策。"""
    next: Literal["researcher", "coder", "FINISH"]
    reason: str


# 初始化 LLM 模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 监督者提示词：指导监督者如何分配任务
SUPERVISOR_PROMPT = (
    "You are a supervisor managing two workers:\n"
    "- researcher: good at finding information, explaining concepts, and analysis\n"
    "- coder: good at writing code, debugging, and technical implementation\n\n"
    "Given the conversation, decide which worker should act next, "
    "or FINISH if the task is complete.\n"
    "Always route to at least one worker before finishing."
)

# 研究员提示词
RESEARCHER_PROMPT = (
    "You are a research assistant. Analyze the request, provide thorough explanations, "
    "gather relevant information, and present your findings clearly. "
    "Keep your response concise but informative."
)

# 程序员提示词
CODER_PROMPT = (
    "You are a coding assistant. Write clean, well-commented code. "
    "If asked to explain code, provide clear explanations with examples. "
    "Keep your response concise and focused on the implementation."
)


def supervisor_node(state: SupervisorState):
    """监督者节点：决定下一个执行的工作者。"""
    # 使用结构化输出获取路由决策
    structured_llm = llm.with_structured_output(RouterOutput)
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    result = structured_llm.invoke(messages)
    return {"next": result.next}


def researcher_node(state: SupervisorState):
    """研究员节点：提供信息分析和解释。"""
    messages = [SystemMessage(content=RESEARCHER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    # 将回复作为 HumanMessage 添加，以便监督者继续处理
    return {
        "messages": [HumanMessage(content=response.content, name="researcher")]
    }


def coder_node(state: SupervisorState):
    """程序员节点：提供代码实现和技术支持。"""
    messages = [SystemMessage(content=CODER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    # 将回复作为 HumanMessage 添加，以便监督者继续处理
    return {
        "messages": [HumanMessage(content=response.content, name="coder")]
    }


def route_supervisor(state: SupervisorState) -> str:
    """路由函数：根据监督者的决策返回下一个节点名称。"""
    next_step = state.get("next", "FINISH")
    if next_step == "FINISH":
        return END  # 任务完成，结束
    return next_step  # 返回工作者名称


def _ensure_messages(input_data: dict) -> dict:
    """确保输入包含正确的消息对象，以兼容 LangServe。"""
    if "messages" not in input_data:
        content = input_data.get("content", str(input_data))
        return {"messages": [HumanMessage(content=content)], "next": ""}

    messages = input_data["messages"]
    if not messages:
        return input_data

    # 将 dict 格式的消息转换为消息对象
    if isinstance(messages[0], dict):
        input_data["messages"] = messages_from_dict(messages)
    # 将字符串列表转换为 HumanMessage 对象
    elif isinstance(messages[0], str):
        input_data["messages"] = [HumanMessage(content=m) for m in messages]

    # 确保 next 字段存在
    if "next" not in input_data:
        input_data["next"] = ""

    return input_data


# 构建 StateGraph
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("coder", coder_node)

# 定义图的结构
builder.add_edge(START, "supervisor")  # 从监督者开始
builder.add_conditional_edges("supervisor", route_supervisor)  # 监督者根据决策路由
builder.add_edge("researcher", "supervisor")  # 研究员完成后返回监督者
builder.add_edge("coder", "supervisor")  # 程序员完成后返回监督者

# 编译图并包装以确保输入格式正确
base_graph = builder.compile()
graph = RunnableLambda(_ensure_messages) | base_graph
