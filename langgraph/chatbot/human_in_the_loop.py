"""
LangGraph Human-in-the-Loop Demo
LangGraph 人机协作演示

Shows how to pause graph execution for human approval before running
sensitive tools, using LangGraph's interrupt() and Command(resume=...).
展示如何在执行敏感工具前暂停图执行以获取人工审批，使用 interrupt() 和 Command(resume=...)。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - interrupt(): pauses the graph and returns control to the caller
    interrupt()：暂停图执行并将控制权返回调用者
  - Command(resume=...): resumes the graph with a user-provided value
    Command(resume=...)：使用用户提供的值恢复图执行
  - MemorySaver: in-memory checkpointer (required for interrupt to work)
    MemorySaver：内存检查点保存器（interrupt 工作的必要条件）
  - thread_id: identifies a conversation thread for checkpointing
    thread_id：标识对话线程，用于检查点管理

Graph structure:
图结构：
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
from langgraph.checkpoint.memory import MemorySaver  # In-memory checkpointer (required for interrupt) / 内存检查点保存器（interrupt 必需）
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt  # interrupt: pause graph; Command: resume execution / interrupt: 暂停图执行; Command: 恢复执行

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define Tools - split into safe and sensitive categories
# 1. 定义工具 - 按安全和敏感类别划分
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time. / 获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient. (Simulated) / 向收件人发送邮件。（模拟）"""
    return f"Email sent to {to} with subject '{subject}'"


@tool
def delete_file(filename: str) -> str:
    """Delete a file from the system. (Simulated) / 从系统中删除文件。（模拟）"""
    return f"File '{filename}' has been deleted."


# Categorize tools by risk level / 按风险等级分类工具
safe_tools = [get_current_time]  # Safe tools: no approval needed / 安全工具：无需审批
sensitive_tools = [send_email, delete_file]  # Sensitive tools: require human approval / 敏感工具：需要人工审批
all_tools = safe_tools + sensitive_tools
sensitive_tool_names = {t.name for t in sensitive_tools}  # Set of sensitive tool names for quick lookup / 敏感工具名称集合，用于快速查找

# ---------------------------------------------------------------------------
# 2. LLM setup
# 2. LLM 配置
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# ---------------------------------------------------------------------------
# 3. Graph Nodes
# 3. 图节点
# ---------------------------------------------------------------------------


def chatbot_node(state: MessagesState):
    """Call the LLM. It may respond with text or tool_calls. / 调用 LLM，可能返回文本或工具调用。"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def human_approval_node(state: MessagesState):
    """If the LLM requested sensitive tools, pause for human approval.
    如果 LLM 请求了敏感工具，暂停等待人工审批。

    interrupt() saves the graph state and returns control to the caller.
    When the caller resumes with Command(resume=value), this function
    continues and `approval` receives that value.
    interrupt() 保存图状态并将控制权返回调用者。
    当调用者使用 Command(resume=value) 恢复时，此函数继续执行，`approval` 接收该值。
    """
    last_message = state["messages"][-1]
    sensitive_calls = [
        tc for tc in last_message.tool_calls if tc["name"] in sensitive_tool_names
    ]

    if sensitive_calls:
        descriptions = "\n".join(
            f"  - {tc['name']}({tc['args']})" for tc in sensitive_calls
        )
        # --- This is where the graph PAUSES / 图在此处暂停，等待人工审批 ---
        approval = interrupt(  # interrupt() saves state and returns control to caller / interrupt() 保存状态并将控制权返回调用者
            f"The agent wants to use sensitive tools:\n{descriptions}\n"
            "Do you approve? (yes/no)"
        )

        if approval.lower() not in ("yes", "y"):  # User denied / 用户拒绝
            return {
                "messages": [AIMessage(content="Action cancelled by user.")]
            }

    return {}  # Approved or no sensitive tools - pass through unchanged / 已批准或无敏感工具 - 不修改状态，直接通过


tool_node = ToolNode(all_tools)

# ---------------------------------------------------------------------------
# 4. Routing Functions
# 4. 路由函数
# ---------------------------------------------------------------------------


def should_use_tools(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "approval"
    return END


def after_approval(state: MessagesState) -> str:
    """After approval node: route to tools if still has tool_calls, else END. / 审批节点后：如果仍有工具调用则路由到工具节点，否则结束。"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# 5. Build the Graph
# 5. 构建图
# ---------------------------------------------------------------------------

graph = StateGraph(MessagesState)

graph.add_node("chatbot", chatbot_node)
graph.add_node("approval", human_approval_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", should_use_tools)
graph.add_conditional_edges("approval", after_approval)
graph.add_edge("tools", "chatbot")

# MemorySaver is REQUIRED for interrupt() to work (needs to save/restore state)
# MemorySaver 是 interrupt() 工作的必要条件（需要保存/恢复状态）
memory = MemorySaver()
app = graph.compile(checkpointer=memory)  # Compile graph with checkpointing / 编译图并启用检查点

# ---------------------------------------------------------------------------
# 6. Interactive CLI
# 6. 交互式命令行
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
    config = {"configurable": {"thread_id": thread_id}}  # Thread config for checkpointing / 线程配置，用于检查点

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # Check if the graph was interrupted (pending nodes)
        # 检查图是否被中断（是否有待执行的节点）
        state = app.get_state(config)
        while state.next:  # state.next is non-empty means graph was paused by interrupt() / state.next 非空表示图被 interrupt() 暂停了
            # Show the interrupt message to the user / 获取中断时传递的消息
            interrupt_value = state.tasks[0].interrupts[0].value
            print(f"\n[Approval Required] {interrupt_value}")
            approval = input("Your decision: ").strip()

            # Resume the graph with the user's decision / 用用户决定恢复图执行
            result = app.invoke(Command(resume=approval), config=config)
            state = app.get_state(config)  # Check again for more interrupts / 再次检查是否还有中断

        ai_message = result["messages"][-1]
        print(f"\nAssistant: {ai_message.content}\n")


if __name__ == "__main__":
    main()
