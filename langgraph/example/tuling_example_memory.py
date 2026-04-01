"""
LangGraph 记忆功能示例：基于CheckPointer的短期记忆

演示如何使用 InMemorySaver 作为检查点存储，使Agent在同一个线程（thread_id）内
保持对话上下文，实现多轮对话的短期记忆功能。
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# 创建内存检查点存储，用于在同一线程内保持对话上下文
checkpointer = InMemorySaver()


def get_weather(city: str) -> str:
    """获取某个城市的天气"""
    return f"城市：{city}, 天气一直都是晴天！"

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

print(cs_response)

# 第二轮对话：使用相同的thread_id继续对话，Agent能记住上下文
bj_response = agent.invoke(
    {"messages": [{"role": "user", "content": "北京呢？"}]},
    config
)

print(bj_response)