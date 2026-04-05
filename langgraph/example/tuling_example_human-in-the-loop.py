"""
LangGraph Human-in-the-Loop 示例：酒店预订Agent

演示如何使用 LangGraph 的 interrupt/Command 机制实现人机交互审批流程。
当Agent调用敏感工具（如预订酒店）时，会暂停执行并等待人工审批，
用户可以选择"OK"直接确认，或"edit"修改参数后再继续执行。

核心组件：
- interrupt(): 暂停Agent执行，向用户发送审批请求
- Command(resume=...): 传递用户的审批结果，恢复Agent执行
- InMemorySaver: 检查点存储，用于在中断/恢复之间保持Agent状态
"""

# 导入环境变量加载工具
from dotenv import load_dotenv
# 导入工具装饰器，用于将函数注册为Agent可调用的工具
from langchain_core.tools import tool
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入内存检查点存储器，用于中断/恢复之间保持状态
from langgraph.checkpoint.memory import InMemorySaver
# 导入中断和命令类型：interrupt用于暂停执行，Command用于恢复执行
from langgraph.types import interrupt, Command
# 导入预构建的ReAct Agent创建函数
from langgraph.prebuilt import create_react_agent

# 加载 .env 文件中的环境变量（如 OPENAI_API_KEY）
load_dotenv()
# 初始化大语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# return_direct=True表示工具的返回值直接作为Agent的最终回复，不再经过LLM处理
@tool(return_direct=True)
def book_hotel(hotel_name: str):
    """预定宾馆

    这是一个敏感操作工具，执行前需要人工审批。
    使用 interrupt() 暂停执行流程，等待用户确认或修改参数。

    支持的审批响应类型：
    - {"type": "OK"}: 用户同意，使用原始参数继续执行
    - {"type": "edit", "args": {"hotel_name": "新酒店名"}}: 用户修改参数后继续执行
    """
    # 暂停Agent，向用户展示即将执行的操作及参数，等待审批
    response = interrupt(
        f"正准备执行'book_hotel'工具预定宾馆，相关参数名： {{'hotel_name': {hotel_name}}}。"
        "请选择OK，表示同意，或者选择edit，提出补充意见。"
    )

    # 根据用户的审批结果处理
    # 判断用户的审批类型并做相应处理
    if response["type"] == "OK":
        pass  # 用户确认，使用原始参数
    elif response["type"] == "edit":
        hotel_name = response["args"]["hotel_name"]  # 使用用户修改后的参数
    else:
        # 未知的响应类型，抛出异常
        raise ValueError(f"Unknown response type: {response['type']}")

    return f"成功在 {hotel_name} 预定了一个房间。"


# InMemorySaver 作为检查点存储，确保 interrupt/resume 之间Agent状态不丢失
checkpointer = InMemorySaver()

# 创建 ReAct Agent，绑定工具和检查点存储
agent = create_react_agent(
    model=llm,
    tools=[book_hotel],
    checkpointer=checkpointer,
)

# thread_id 用于标识对话线程，检查点按线程隔离
config = {
    "configurable": {
        "thread_id": "1"
    }
}

if __name__ == "__main__":
    # 第一阶段：发送用户请求，Agent会调用 book_hotel 工具并在 interrupt() 处暂停
    # 使用stream流式输出，逐步打印Agent的执行过程
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "帮我在图灵宾馆预定一个房间"}]},
        config
    ):
        print(chunk)
        print("\n")

    # 第二阶段：模拟用户审批，使用 Command(resume=...) 恢复Agent执行
    # 这里选择了"edit"模式，将酒店名修改为"三号宾馆"
    for chunk in agent.stream(
       # Command(resume={"type": "OK"}),  # 如果直接同意，取消此行注释
        Command(resume={"type": "edit", "args": {"hotel_name": "三号宾馆"}}),
        config
    ):
        print(chunk)
        # 打印工具执行的最终结果
        print(chunk['tools']['messages'][-1].content)
        print("\n")
