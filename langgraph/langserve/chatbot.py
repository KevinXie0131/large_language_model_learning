"""聊天机器人 Agent，使用 StateGraph 和模拟工具。"""

from datetime import datetime
import ast
import operator

from langchain_core.messages import HumanMessage, messages_from_dict
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气。"""
    # 模拟天气数据
    mock_weather = {
        "tokyo": "☀️ 22°C, sunny with light breeze",
        "london": "🌧️ 14°C, overcast with light rain",
        "new york": "⛅ 18°C, partly cloudy",
        "paris": "🌤️ 20°C, mostly sunny",
        "sydney": "☀️ 26°C, clear skies",
    }
    return mock_weather.get(city.lower(), f"🌡️ 20°C, pleasant weather in {city}")


@tool
def get_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。例如：'2 + 3 * 4'。"""
    # 允许的字符集合，用于安全验证
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only numbers and +-*/.() are allowed"
    try:
        # 使用 AST 进行安全的数学表达式求值（替代危险的 eval）
        operators_map = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def _eval_node(node: ast.AST):
            """递归解析 AST 节点并计算结果。"""
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Only numbers are allowed")
            if isinstance(node, ast.BinOp):
                # 二元运算（如 2 + 3）
                op_func = operators_map.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"Unsupported operator: {type(node.op)}")
                return op_func(_eval_node(node.left), _eval_node(node.right))
            if isinstance(node, ast.UnaryOp):
                # 一元运算（如 -5）
                op_func = operators_map.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"Unsupported operator: {type(node.op)}")
                return op_func(_eval_node(node.operand))
            raise ValueError("Invalid expression")

        # 解析表达式并计算
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _ensure_messages(input_data: dict) -> dict:
    """确保输入包含正确的消息对象，以兼容 LangServe。"""
    if "messages" not in input_data:
        content = input_data.get("content", str(input_data))
        return {"messages": [HumanMessage(content=content)]}

    messages = input_data["messages"]
    if not messages:
        return input_data

    # 将 dict 格式的消息转换为消息对象
    if isinstance(messages[0], dict):
        input_data["messages"] = messages_from_dict(messages)
    # 将字符串列表转换为 HumanMessage 对象
    elif isinstance(messages[0], str):
        input_data["messages"] = [HumanMessage(content=m) for m in messages]

    return input_data


# 定义工具列表
tools = [get_weather, get_time, calculator]
# 初始化 LLM 并绑定工具
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


def chatbot_node(state: MessagesState):
    """聊天机器人节点：调用 LLM 处理消息。"""
    return {"messages": [llm.invoke(state["messages"])]}


def should_use_tools(state: MessagesState) -> str:
    """条件路由函数：判断是否需要调用工具。"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # 有工具调用，路由到工具节点
    return END  # 无工具调用，结束对话


# 构建 StateGraph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", should_use_tools)
builder.add_edge("tools", "chatbot")

# 编译图并包装以确保输入格式正确
base_graph = builder.compile()
graph = RunnableLambda(_ensure_messages) | base_graph
