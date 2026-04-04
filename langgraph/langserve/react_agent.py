"""ReAct Agent，使用 create_react_agent 和模拟工具。"""

from datetime import datetime
import ast
import operator

from langchain_core.messages import HumanMessage, messages_from_dict
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def search_web(query: str) -> str:
    """搜索网络获取主题信息。"""
    # 模拟搜索结果数据
    mock_results = {
        "langgraph": (
            "LangGraph is a framework for building stateful, multi-actor "
            "applications with LLMs. It extends LangChain with cyclic graph "
            "support for complex agent workflows."
        ),
        "langserve": (
            "LangServe helps deploy LangChain runnables and chains as REST APIs. "
            "It provides /invoke, /stream, and /playground endpoints automatically."
        ),
        "python": (
            "Python is a high-level programming language known for its simplicity. "
            "Python 3.12 is the latest stable release with improved performance."
        ),
    }
    # 遍历模拟结果，查找匹配的关键词
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': No specific results found. This is a mock search tool for demonstration purposes."


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


# 初始化 LLM 模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 创建 ReAct Agent 图
base_graph = create_react_agent(llm, tools=[search_web, get_time, calculator])
# 包装图以确保输入格式正确
graph = RunnableLambda(_ensure_messages) | base_graph
