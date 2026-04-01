from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


checkpointer = InMemorySaver()

# Trae: 解释代码 | 注释代码 | 生成单测 | 探索 IDE | X
def get_weather(city: str) -> str:
    """获取某个城市的天气"""
    return f"城市：{city}, 天气一直都是晴天！"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=checkpointer
)

# Run the agent
config = {
    "configurable": {
        "thread_id": "1"
    }
}

cs_response = agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字是小智"}]},
    config
)

print(cs_response)
config = {
    "configurable": {
        "thread_id": "2"
    }
}
# Continue the conversation using the same thread_id
bj_response = agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字是?"}]},
    config
)

print(bj_response)

config = {
    "configurable": {
        "thread_id": "1"
    }
}
# Continue the conversation using the same thread_id
cj_response = agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字是?"}]},
    config
)
print('#'*40)
print(cj_response)