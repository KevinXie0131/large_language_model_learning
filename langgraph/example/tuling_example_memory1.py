"""
LangGraph 记忆功能示例：线程隔离验证

演示不同 thread_id 之间的对话上下文是隔离的。
thread_id="1" 中告诉Agent自己的名字，在 thread_id="2" 中Agent无法得知，
但回到 thread_id="1" 时Agent仍然记得。
"""

# 导入内存检查点存储器
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

# 创建内存检查点存储
checkpointer = InMemorySaver()


def get_weather(city: str) -> str:
    """获取某个城市的天气"""
    return f"城市：{city}, 天气一直都是晴天！"

# 初始化大语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创建ReAct Agent
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=checkpointer
)

# 第一个线程：thread_id="1"
config = {
    "configurable": {
        "thread_id": "1"
    }
}

# 在线程1中告诉Agent自己的名字
cs_response = agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字是小智"}]},
    config
)

# 打印线程1的响应
print(cs_response)

# 切换到线程2（新的对话上下文，无法访问线程1的记忆）
config = {
    "configurable": {
        "thread_id": "2"
    }
}
# 在线程2中询问名字（Agent不知道，因为线程隔离）
bj_response = agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字是?"}]},
    config
)

# 打印线程2的响应（预期不知道用户名字）
print(bj_response)

# 切换回线程1（恢复之前的对话上下文）
config = {
    "configurable": {
        "thread_id": "1"
    }
}
# 在线程1中再次询问名字（Agent应该记得是"小智"）
cj_response = agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字是?"}]},
    config
)
# 打印分隔线
print('#'*40)
# 打印线程1的响应（预期能记住用户名字是"小智"）
print(cj_response)