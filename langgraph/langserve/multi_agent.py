"""Multi-agent supervisor that routes tasks to researcher and coder agents."""

from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


class RouterOutput(BaseModel):
    next: Literal["researcher", "coder", "FINISH"]
    reason: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SUPERVISOR_PROMPT = (
    "You are a supervisor managing two workers:\n"
    "- researcher: good at finding information, explaining concepts, and analysis\n"
    "- coder: good at writing code, debugging, and technical implementation\n\n"
    "Given the conversation, decide which worker should act next, "
    "or FINISH if the task is complete.\n"
    "Always route to at least one worker before finishing."
)

RESEARCHER_PROMPT = (
    "You are a research assistant. Analyze the request, provide thorough explanations, "
    "gather relevant information, and present your findings clearly. "
    "Keep your response concise but informative."
)

CODER_PROMPT = (
    "You are a coding assistant. Write clean, well-commented code. "
    "If asked to explain code, provide clear explanations with examples. "
    "Keep your response concise and focused on the implementation."
)


def supervisor_node(state: SupervisorState):
    structured_llm = llm.with_structured_output(RouterOutput)
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    result = structured_llm.invoke(messages)
    return {"next": result.next}


def researcher_node(state: SupervisorState):
    messages = [SystemMessage(content=RESEARCHER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {
        "messages": [HumanMessage(content=response.content, name="researcher")]
    }


def coder_node(state: SupervisorState):
    messages = [SystemMessage(content=CODER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {
        "messages": [HumanMessage(content=response.content, name="coder")]
    }


def route_supervisor(state: SupervisorState) -> str:
    next_step = state.get("next", "FINISH")
    if next_step == "FINISH":
        return END
    return next_step


builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("coder", coder_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor)
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

graph = builder.compile()
