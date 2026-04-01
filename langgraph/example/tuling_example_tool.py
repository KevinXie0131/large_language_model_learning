from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# 定义工具 return_direct=True 表示直接返回工具的结果
@tool("devide_tool", return_direct=True)
def devide(a : int, b : int) -> float:
    """计算两个整数的除法。
    Args:
        a (int): 除数
        b (int): 被除数"""
    # 自定义错误
    if b == 1:
        raise ValueError("除数不能为1")
    return a/b

print(devide.name)
print(devide.description)
print(devide.args)

# 定义工具调用错误处理函数
def handle_tool_error(error: Exception) -> str:
    """处理工具调用错误。
    Args:
        error (Exception): 工具调用错误"""
    if isinstance(error, ValueError):
        return "除数为1没有意义，请重新输入一个除数和被除数。"
    elif isinstance(error, ZeroDivisionError):
        return "除数不能为0，请重新输入一个除数和被除数。"
    return f"工具调用错误: {error}"

tool_node = ToolNode(
    [devide],
    handle_tool_errors=handle_tool_error
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent_with_error_handler = create_react_agent(
    model=llm,
    tools=tool_node
)

result = agent_with_error_handler.invoke({"messages":[{"role":"user","content":"10除以1等于多少？"}]})
# 打印最后的返回结果
print(result["messages"][-1].content)