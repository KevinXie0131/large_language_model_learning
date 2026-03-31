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
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define the routing schema
#    The supervisor uses structured output to pick the next worker.
# ---------------------------------------------------------------------------

WORKERS = ["researcher", "coder"]


class RouterOutput(BaseModel):
    """The supervisor's routing decision."""

    next: Literal["researcher", "coder", "FINISH"]
    reason: str


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
    structured_llm = llm.with_structured_output(RouterOutput)
    response = structured_llm.invoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM)] + state["messages"]
    )
    if response.next == "FINISH":
        return {"messages": []}
    return {"messages": [], "next": response.next}


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
        [SystemMessage(content=RESEARCHER_SYSTEM)] + state["messages"]
    )
    return {
        "messages": [
            HumanMessage(content=response.content, name="researcher")
        ]
    }


def coder_node(state: MessagesState):
    """Coder worker - writes and explains code."""
    response = llm.invoke(
        [SystemMessage(content=CODER_SYSTEM)] + state["messages"]
    )
    return {
        "messages": [HumanMessage(content=response.content, name="coder")]
    }


# ---------------------------------------------------------------------------
# 5. Routing Function
# ---------------------------------------------------------------------------


def route_supervisor(state: MessagesState) -> str:
    """Route based on the supervisor's decision stored in state."""
    next_worker = state.get("next", "FINISH")
    if next_worker == "FINISH":
        return END
    return next_worker


# ---------------------------------------------------------------------------
# 6. Build the Graph
# ---------------------------------------------------------------------------

# Use a custom state that includes a "next" field for routing
from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


graph = StateGraph(SupervisorState)

# Add all nodes
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", coder_node)

# Entry point
graph.add_edge(START, "supervisor")

# Supervisor routes to workers or END
graph.add_conditional_edges("supervisor", route_supervisor)

# Workers always report back to supervisor
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
        events = app.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="updates",
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
