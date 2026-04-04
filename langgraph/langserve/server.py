"""LangServe 服务器，将 4 个 LangGraph Agent 示例暴露为 REST API。"""

from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量

from fastapi import FastAPI  # noqa: E402
from langserve import add_routes  # noqa: E402
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableConfig

from chatbot import graph as chatbot_graph  # noqa: E402
from react_agent import graph as react_agent_graph  # noqa: E402
from rag_agent import graph as rag_agent_graph  # noqa: E402
from multi_agent import graph as multi_agent_graph  # noqa: E402

# 创建 FastAPI 应用实例
app = FastAPI(
    title="LangGraph Agents API",
    version="1.0.0",
    description="4 LangGraph agent examples served via LangServe",
)


class ChatInput(BaseModel):
    """聊天输入模型，用于 LangServe Playground 的简单文本输入。"""
    message: str


def wrap_for_playground(graph, extra_state=None):
    """包装 graph 以接受来自 LangServe Playground 的简单文本输入。"""
    def process(input_data, config: RunnableConfig = None):
        # 兼容 dict 和 ChatInput 对象两种输入格式
        if isinstance(input_data, dict):
            message = input_data.get("message", str(input_data))
        elif hasattr(input_data, "message"):
            message = input_data.message
        else:
            message = str(input_data)
        
        # 构建 LangGraph 需要的状态格式
        state = {"messages": [HumanMessage(content=message)]}
        if extra_state:
            state.update(extra_state)
        # 确保 thread_id 已设置，用于会话持久化
        if config is None:
            config = {}
        if "configurable" not in config:
            config["configurable"] = {}
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default"
        return state
    return RunnableLambda(process) | graph


@app.get("/")
async def root():
    """根路径，返回 API 信息和可用端点列表。"""
    return {
        "message": "LangGraph Agents API",
        "endpoints": {
            "/chatbot": "带模拟工具的聊天机器人（天气、时间、计算器）",
            "/react-agent": "带模拟工具的 ReAct Agent（搜索、时间、计算器）",
            "/rag-agent": "带 LangGraph 知识库的 RAG Agent",
            "/multi-agent": "带研究员和程序员的多 Agent 监督系统",
        },
        "playground": {
            "/chatbot/playground": "聊天机器人测试界面",
            "/react-agent/playground": "ReAct Agent 测试界面",
            "/rag-agent/playground": "RAG Agent 测试界面",
            "/multi-agent/playground": "多 Agent 测试界面",
        },
        "docs": "/docs",
    }


# 注册 4 个 Agent 的路由，使用 ChatInput 作为输入类型
add_routes(app, wrap_for_playground(chatbot_graph), path="/chatbot", input_type=ChatInput)
add_routes(app, wrap_for_playground(react_agent_graph), path="/react-agent", input_type=ChatInput)
add_routes(app, wrap_for_playground(rag_agent_graph), path="/rag-agent", input_type=ChatInput)
add_routes(app, wrap_for_playground(multi_agent_graph, extra_state={"next": ""}), path="/multi-agent", input_type=ChatInput)

if __name__ == "__main__":
    import uvicorn

    # 启动 Uvicorn 服务器，监听所有接口的 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
