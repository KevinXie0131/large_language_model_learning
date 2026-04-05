"""
LangGraph 记忆功能示例：基于CheckPointer的短期记忆

演示如何使用 InMemorySaver 作为检查点存储，使Agent在同一个线程（thread_id）内
保持对话上下文，实现多轮对话的短期记忆功能。
"""

# 导入内存检查点存储器，用于保存对话状态
from langgraph.checkpoint.memory import InMemorySaver
# 导入预构建的ReAct Agent创建函数
from langgraph.prebuilt import create_react_agent
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI

import os
from datetime import datetime
# 导入环境变量加载工具
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 创建内存检查点存储，用于在同一线程内保持对话上下文
checkpointer = InMemorySaver()


def get_weather(city: str) -> str:
    """获取某个城市的天气"""
    return f"城市：{city}, 天气一直都是晴天！"

# 初始化大语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创建ReAct Agent，绑定工具和检查点存储
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=checkpointer
)

# thread_id 标识对话线程，同一线程内的消息共享上下文
config = {
    "configurable": {
        "thread_id": "1"
    }
}

# 第一轮对话：询问长沙天气
cs_response = agent.invoke(
    {"messages": [{"role": "user", "content": "长沙天气怎么样？"}]},
    config
)

# 打印第一轮对话的完整响应
print(cs_response)

# 第二轮对话：使用相同的thread_id继续对话，Agent能记住上下文
bj_response = agent.invoke(
    {"messages": [{"role": "user", "content": "北京呢？"}]},
    config
)

# 打印第二轮对话的完整响应
print(bj_response)