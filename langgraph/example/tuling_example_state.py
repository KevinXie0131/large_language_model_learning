"""
LangGraph 自定义状态示例：通过InjectedState在工具中访问Agent状态

演示如何扩展 AgentState 添加自定义字段（如 user_id），
并通过 InjectedState 注入机制让工具函数能够直接访问Agent的完整状态。
"""

from typing import Annotated

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 自定义状态类，在默认AgentState基础上添加 user_id 字段
class CustomState(AgentState):
    user_id: str


# return_direct=True 表示工具的返回值直接作为Agent的最终回复
@tool(return_direct=True)
def get_user_info(
    # 使用 InjectedState 注入完整的Agent状态，工具可以直接访问 state["user_id"]
    state: Annotated[CustomState, InjectedState],
) -> str:
    """查询用户信息。"""
    user_id = state["user_id"]
    return "user_123用户的姓名：楼兰。" if user_id == "user_123" else "未知用户"


# 创建Agent时指定自定义状态类
agent = create_react_agent(
    model=llm,
    tools=[get_user_info],
    state_schema=CustomState,  # 使用自定义状态
)

if __name__ == "__main__":
    # 调用时传入自定义状态字段 user_id
    result = agent.invoke({
        "messages": "查询用户信息",
        "user_id": "user_123",
    })
    print(result["messages"][-1].content)
