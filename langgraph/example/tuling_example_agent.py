import logging

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def book_hotel(hotel_name: str):
    """Book a hotel"""
    logger.info(f"预订酒店: {hotel_name}")
    return f"已成功预订入住于 {hotel_name}."


def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    logger.info(f"预订航班: {from_airport} -> {to_airport}")
    return f"已成功预订从 {from_airport} 到 {to_airport}的航班."


flight_assistant = create_react_agent(
    model=llm,
    tools=[book_flight],
    prompt="你是一个航班预订助手",
    name="flight_assistant",
)

hotel_assistant = create_react_agent(
    model=llm,
    tools=[book_hotel],
    prompt="你是一个酒店预订助手",
    name="hotel_assistant",
)

supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=llm,
    prompt=(
        "你管理一个酒店预订助手和一个"
        "航班预订助手。将工作分配给它们。"
    ),
).compile()

if __name__ == "__main__":
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
        for node, state in chunk.items():
            for msg in state.get("messages", []):
                if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                    print(f"[{node}] {msg.pretty_print()}")

