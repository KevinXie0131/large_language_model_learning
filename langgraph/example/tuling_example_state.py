from typing import Annotated

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class CustomState(AgentState):
    user_id: str


@tool(return_direct=True)
def get_user_info(
    state: Annotated[CustomState, InjectedState],
) -> str:
    """查询用户信息。"""
    user_id = state["user_id"]
    return "user_123用户的姓名：楼兰。" if user_id == "user_123" else "未知用户"


agent = create_react_agent(
    model=llm,
    tools=[get_user_info],
    state_schema=CustomState,
)

if __name__ == "__main__":
    result = agent.invoke({
        "messages": "查询用户信息",
        "user_id": "user_123",
    })
    print(result["messages"][-1].content)
