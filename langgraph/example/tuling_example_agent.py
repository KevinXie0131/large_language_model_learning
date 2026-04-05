"""
LangGraph 多Agent协作示例：Supervisor模式

演示如何使用 langgraph_supervisor 创建一个主管Agent（Supervisor），
由主管Agent协调多个子Agent（航班助手、酒店助手）协同完成复杂任务。
"""

# 导入日志模块，用于记录运行时信息
import logging

# 导入环境变量加载工具
from dotenv import load_dotenv
# 导入消息类型：AI消息、用户消息、工具消息
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入预构建的ReAct Agent创建函数
from langgraph.prebuilt import create_react_agent
# 导入Supervisor（主管）Agent创建函数，用于多Agent协作
from langgraph_supervisor import create_supervisor

# 加载环境变量（如 OPENAI_API_KEY）
load_dotenv()
# 配置日志级别为INFO，输出运行时信息
logging.basicConfig(level=logging.INFO)
# 获取当前模块的日志记录器
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
    agents=[flight_assistant, hotel_assistant],  # 注册所有子Agent
    model=llm,  # Supervisor使用的大语言模型
    prompt=(
        "你管理一个酒店预订助手和一个"
        "航班预订助手。将工作分配给它们。"
    ),
).compile()  # 编译Supervisor图，使其可以执行

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
                # 过滤并打印用户消息、AI消息和工具消息
                if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                    print(f"[{node}] {msg.pretty_print()}")

