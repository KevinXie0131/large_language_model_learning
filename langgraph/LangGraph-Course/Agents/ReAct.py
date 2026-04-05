# 导入类型提示相关模块
from typing import Annotated, Sequence, TypedDict
# 导入环境变量加载工具
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入工具装饰器
from langchain_core.tools import tool
# 导入消息累加器函数
from langgraph.graph.message import add_messages
# 导入状态图和终点
from langgraph.graph import StateGraph, END
# 导入预构建的工具节点
from langgraph.prebuilt import ToolNode


# 加载环境变量
load_dotenv()

# 定义代理状态，使用add_messages自动累加消息
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 定义加法工具
@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""

    return a + b

# 定义减法工具
@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

# 定义乘法工具
@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

# 将所有工具放入列表
tools = [add, subtract, multiply]

# 初始化模型并绑定工具（让模型知道可以调用哪些工具）
model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)


# 模型调用节点：发送系统提示和消息给LLM
def model_call(state:AgentState) -> AgentState:
    # 定义系统提示词
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    # 调用模型，将系统提示和用户消息一起发送
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# 条件判断函数：决定是否继续调用工具
def should_continue(state: AgentState):
    messages = state["messages"]
    # 获取最后一条消息
    last_message = messages[-1]
    # 如果没有工具调用，则结束；否则继续
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


# 创建状态图
graph = StateGraph(AgentState)
# 添加代理节点（LLM调用）
graph.add_node("our_agent", model_call)


# 创建工具节点（执行工具调用）
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

# 设置入口点为代理节点
graph.set_entry_point("our_agent")

# 添加条件边：根据should_continue的返回值决定下一步
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",  # 有工具调用则转到工具节点
        "end": END,  # 无工具调用则结束
    },
)

# 工具执行完毕后返回代理节点（形成ReAct循环）
graph.add_edge("tools", "our_agent")

# 编译图
app = graph.compile()

# 打印流式输出的辅助函数
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            # 使用美化打印输出消息
            message.pretty_print()

# 定义用户输入（包含多步工具调用的复杂任务）
inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
# 以流式模式运行代理并打印结果
print_stream(app.stream(inputs, stream_mode="values"))
