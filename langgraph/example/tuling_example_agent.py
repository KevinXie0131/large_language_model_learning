"""
LangGraph 多Agent协作示例：Supervisor模式

演示如何使用 langgraph_supervisor 创建一个主管Agent（Supervisor），
由主管Agent协调多个子Agent（航班助手、酒店助手）协同完成复杂任务。
"""

import logging

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# 加载环境变量（如 OPENAI_API_KEY）
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 初始化大语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 定义酒店预订工具
def book_hotel(hotel_name: str):
    """Book a hotel"""
    logger.info(f"预订酒店: {hotel_name}")
    return f"已成功预订入住于 {hotel_name}."


# 定义航班预订工具
def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    logger.info(f"预订航班: {from_airport} -> {to_airport}")
    return f"已成功预订从 {from_airport} 到 {to_airport}的航班."


# 创建航班预订子Agent（ReAct模式）
flight_assistant = create_react_agent(
    model=llm,
    tools=[book_flight],
    prompt="你是一个航班预订助手",
    name="flight_assistant",
)

# 创建酒店预订子Agent（ReAct模式）
hotel_assistant = create_react_agent(
    model=llm,
    tools=[book_hotel],
    prompt="你是一个酒店预订助手",
    name="hotel_assistant",
)

# 创建主管Agent（Supervisor），负责协调子Agent之间的任务分配
supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=llm,
    prompt=(
        "你管理一个酒店预订助手和一个"
        "航班预订助手。将工作分配给它们。"
    ),
).compile()

if __name__ == "__main__":
    # 以流式方式运行Supervisor，发送用户请求
    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "预订一张从波士顿（BOS）到纽约肯尼迪机场（JFK）的航班，并预订麦基特里克酒店的住宿。",
                }
            ]
        }
    ):
        # 遍历每个节点的输出，打印消息内容
        for node, state in chunk.items():
            for msg in state.get("messages", []):
                if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                    print(f"[{node}] {msg.pretty_print()}")

