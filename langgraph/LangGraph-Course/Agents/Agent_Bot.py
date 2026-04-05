# 导入类型提示相关模块
from typing import TypedDict, List
# 导入人类消息类
from langchain_core.messages import HumanMessage
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入状态图、起点和终点
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values

# 加载环境变量（如API密钥）
load_dotenv()

# 定义代理状态的数据结构，包含消息列表
class AgentState(TypedDict):
    messages: List[HumanMessage]

# 初始化GPT-4o大语言模型
llm = ChatOpenAI(model="gpt-4o")

# 处理函数：将用户消息发送给LLM并打印回复
def process(state: AgentState) -> AgentState:
    # 调用LLM处理消息
    response = llm.invoke(state["messages"])
    # 打印AI的回复内容
    print(f"\nAI: {response.content}")
    return state

# 创建状态图
graph = StateGraph(AgentState)
# 添加处理节点
graph.add_node("process", process)
# 设置从起点到处理节点的边
graph.add_edge(START, "process")
# 设置从处理节点到终点的边
graph.add_edge("process", END)
# 编译图为可执行的代理
agent = graph.compile()

# 获取用户输入
user_input = input("Enter: ")
# 循环对话，直到用户输入"exit"退出
while user_input != "exit":
    # 调用代理处理用户输入
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
