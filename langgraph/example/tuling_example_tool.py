"""
LangGraph 工具与错误处理示例

演示如何定义自定义工具（带 return_direct 属性）、
使用 ToolNode 封装工具节点、以及自定义工具调用错误处理函数。
"""

# 导入工具装饰器，用于将函数注册为Agent可调用的工具
from langchain_core.tools import tool
# 导入工具节点，用于封装工具并支持自定义错误处理
from langgraph.prebuilt import ToolNode
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入预构建的ReAct Agent创建函数
from langgraph.prebuilt import create_react_agent
import os
from datetime import datetime
# 导入环境变量加载工具
from dotenv import load_dotenv

# 加载.env文件中的环境变量
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
    # 执行除法运算
    return a/b

# 查看工具的元信息（名称、描述、参数定义）
print(devide.name)  # 打印工具名称
print(devide.description)  # 打印工具描述
print(devide.args)  # 打印工具参数结构


# 自定义工具调用错误处理函数，根据异常类型返回不同的友好提示
def handle_tool_error(error: Exception) -> str:
    """处理工具调用错误。
    Args:
        error (Exception): 工具调用错误"""
    # 处理自定义的ValueError异常
    if isinstance(error, ValueError):
        return "除数为1没有意义，请重新输入一个除数和被除数。"
    # 处理除零异常
    elif isinstance(error, ZeroDivisionError):
        return "除数不能为0，请重新输入一个除数和被除数。"
    # 处理其他未知异常
    return f"工具调用错误: {error}"

# 创建工具节点，绑定自定义错误处理
tool_node = ToolNode(
    [devide],
    handle_tool_errors=handle_tool_error  # 指定错误处理函数
)

# 初始化大语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创建Agent，使用包含错误处理的ToolNode（而非直接传入工具列表）
agent_with_error_handler = create_react_agent(
    model=llm,
    tools=tool_node
)

# 调用Agent处理用户的除法请求（会触发自定义错误处理）
result = agent_with_error_handler.invoke({"messages":[{"role":"user","content":"10除以1等于多少？"}]})
# 打印最后的返回结果（工具会抛出自定义错误，Agent会收到友好的错误提示）
print(result["messages"][-1].content)