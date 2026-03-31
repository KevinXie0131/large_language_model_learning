"""
LangGraph Human-in-the-Loop Demo

Shows how to pause graph execution for human approval before running
sensitive tools, using LangGraph's interrupt() and Command(resume=...).

Key LangGraph concepts demonstrated:
  - interrupt(): pauses the graph and returns control to the caller
  - Command(resume=...): resumes the graph with a user-provided value
  - MemorySaver: in-memory checkpointer (required for interrupt to work)
  - thread_id: identifies a conversation thread for checkpointing

Graph structure:
  START → chatbot → (has tool calls?) → approval → (approved?) → tools → chatbot
                       ↘ (no)                        ↘ (denied)
                        END                            END
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define Tools - split into safe and sensitive categories
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient. (Simulated)"""
    return f"Email sent to {to} with subject '{subject}'"


@tool
def delete_file(filename: str) -> str:
    """Delete a file from the system. (Simulated)"""
    return f"File '{filename}' has been deleted."


# Categorize tools by risk level
safe_tools = [get_current_time]
sensitive_tools = [send_email, delete_file]
all_tools = safe_tools + sensitive_tools
sensitive_tool_names = {t.name for t in sensitive_tools}

# ---------------------------------------------------------------------------
# 2. LLM setup
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# ---------------------------------------------------------------------------
# 3. Graph Nodes
# ---------------------------------------------------------------------------


def chatbot_node(state: MessagesState):
    """Call the LLM. It may respond with text or tool_calls."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def human_approval_node(state: MessagesState):
    """If the LLM requested sensitive tools, pause for human approval.

    interrupt() saves the graph state and returns control to the caller.
    When the caller resumes with Command(resume=value), this function
    continues and `approval` receives that value.
    """
    last_message = state["messages"][-1]
    sensitive_calls = [
        tc for tc in last_message.tool_calls if tc["name"] in sensitive_tool_names
    ]

    if sensitive_calls:
        descriptions = "\n".join(
            f"  - {tc['name']}({tc['args']})" for tc in sensitive_calls
        )
        # --- This is where the graph PAUSES ---
        approval = interrupt(
            f"The agent wants to use sensitive tools:\n{descriptions}\n"
            "Do you approve? (yes/no)"
        )

        if approval.lower() not in ("yes", "y"):
            # User denied - replace with cancellation message
            return {
                "messages": [AIMessage(content="Action cancelled by user.")]
            }

    # Approved or no sensitive tools - pass through unchanged
    return {}


tool_node = ToolNode(all_tools)

# ---------------------------------------------------------------------------
# 4. Routing Functions
# ---------------------------------------------------------------------------


def should_use_tools(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "approval"
    return END


def after_approval(state: MessagesState) -> str:
    """After approval node: route to tools if still has tool_calls, else END."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# 5. Build the Graph
# ---------------------------------------------------------------------------

graph = StateGraph(MessagesState)

graph.add_node("chatbot", chatbot_node)
graph.add_node("approval", human_approval_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", should_use_tools)
graph.add_conditional_edges("approval", after_approval)
graph.add_edge("tools", "chatbot")

# MemorySaver is REQUIRED for interrupt() to work
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# 6. Interactive CLI
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Human-in-the-Loop Demo")
    print("  Safe tools: get_current_time")
    print("  Sensitive tools: send_email, delete_file (require approval)")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print("\nTry: 'Send an email to bob@example.com about the meeting'")
    print("Or:  'What time is it?' (no approval needed)\n")

    thread_id = "hitl-thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Invoke the graph
        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # Check if the graph was interrupted (pending nodes)
        state = app.get_state(config)
        while state.next:
            # Show the interrupt message to the user
            interrupt_value = state.tasks[0].interrupts[0].value
            print(f"\n[Approval Required] {interrupt_value}")
            approval = input("Your decision: ").strip()

            # Resume the graph with the user's decision
            result = app.invoke(Command(resume=approval), config=config)
            state = app.get_state(config)

        ai_message = result["messages"][-1]
        print(f"\nAssistant: {ai_message.content}\n")


if __name__ == "__main__":
    main()
