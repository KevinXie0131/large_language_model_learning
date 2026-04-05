"""
LangGraph 自定义状态示例：通过InjectedState在工具中访问Agent状态

演示如何扩展 AgentState 添加自定义字段（如 user_id），
并通过 InjectedState 注入机制让工具函数能够直接访问Agent的完整状态。
"""

# 导入Annotated类型注解，用于工具函数的参数注入
from typing import Annotated

# 导入环境变量加载工具
from dotenv import load_dotenv
# 导入工具装饰器
from langchain_core.tools import tool
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入状态注入工具和ReAct Agent创建函数
from langgraph.prebuilt import InjectedState, create_react_agent
# 导入Agent的默认状态类，用于扩展自定义状态
from langgraph.prebuilt.chat_agent_executor import AgentState

# 加载.env文件中的环境变量
load_dotenv()
# 初始化大语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 自定义状态类，在默认AgentState基础上添加 user_id 字段
class CustomState(AgentState):
    user_id: str  # 新增的用户ID字段，用于在工具中获取用户标识


# return_direct=True 表示工具的返回值直接作为Agent的最终回复
@tool(return_direct=True)
def get_user_info(
    # 使用 InjectedState 注入完整的Agent状态，工具可以直接访问 state["user_id"]
    state: Annotated[CustomState, InjectedState],
) -> str:
    """查询用户信息。"""
    # 从注入的状态中获取用户ID
    user_id = state["user_id"]
    # 根据用户ID返回对应的用户信息
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
        "messages": "查询用户信息",  # 用户的请求消息
        "user_id": "user_123",  # 自定义状态字段，传入用户ID
    })
    # 打印Agent的最终回复
    print(result["messages"][-1].content)
