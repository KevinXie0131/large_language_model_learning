# 导入操作系统模块
import os
# 导入类型提示相关模块
from typing import TypedDict, List, Union
# 导入人类消息和AI消息类
from langchain_core.messages import HumanMessage, AIMessage
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入状态图、起点和终点
from langgraph.graph import StateGraph, START, END
# 导入环境变量加载工具
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 定义代理状态，消息列表可包含人类消息和AI消息
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# 初始化GPT-4o大语言模型
llm = ChatOpenAI(model="gpt-4o")

# 处理函数：调用LLM并将AI回复追加到对话历史中
def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    # 调用LLM获取回复
    response = llm.invoke(state["messages"])

    # 将AI回复添加到消息历史中（实现记忆功能）
    state["messages"].append(AIMessage(content=response.content))
    # 打印AI回复
    print(f"\nAI: {response.content}")
    # 打印当前完整状态（用于调试）
    print("CURRENT STATE: ", state["messages"])

    return state

# 构建状态图
graph = StateGraph(AgentState)
# 添加处理节点
graph.add_node("process", process)
# 连接起点到处理节点
graph.add_edge(START, "process")
# 连接处理节点到终点
graph.add_edge("process", END)
# 编译图为可执行的代理
agent = graph.compile()


# 初始化对话历史列表
conversation_history = []

# 获取用户输入
user_input = input("Enter: ")
# 循环对话，直到用户输入"exit"
while user_input != "exit":
    # 将用户输入添加到对话历史
    conversation_history.append(HumanMessage(content=user_input))
    # 调用代理处理，传入完整对话历史
    result = agent.invoke({"messages": conversation_history})
    # 更新对话历史（包含AI回复）
    conversation_history = result["messages"]
    user_input = input("Enter: ")


# 将对话记录保存到文件
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")

    # 遍历对话历史，区分人类消息和AI消息
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

# 提示用户对话已保存
print("Conversation saved to logging.txt")
