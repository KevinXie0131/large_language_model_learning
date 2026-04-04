# LangServe 完整指南

## 简介

LangServe 是 LangChain 官方提供的部署框架，用于将 LangChain 的 Runnable、Chain 和 LangGraph 图快速部署为 REST API。它基于 FastAPI 构建，能够自动为任何可运行对象生成标准化的 API 端点，包括调用、流式输出、批处理和交互式 Playground 界面。

LangServe 的核心理念是：**一次编写，随处部署**。开发者只需专注于构建 Agent 或 Chain 的逻辑，LangServe 会自动处理 API 路由、请求验证、文档生成和前端界面。

---

## 核心特性

### 1. 自动 API 端点生成

通过 `add_routes()` 函数，LangServe 会为每个 Runnable 自动生成以下端点：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/{path}/invoke` | POST | 单次调用，返回完整结果 |
| `/{path}/stream` | POST | 流式输出（SSE），适合实时响应 |
| `/{path}/batch` | POST | 批量处理多个输入 |
| `/{path}/astream_events` | POST | 流式事件，可观察中间步骤 |
| `/{path}/playground` | GET | 交互式 Web 测试界面 |

### 2. 内置 OpenAPI 文档

启动服务后，访问 `/docs` 即可查看自动生成的 Swagger UI 文档，包含所有端点的请求/响应格式、参数说明和在线测试功能。

### 3. 交互式 Playground

每个部署的 Runnable 都自带一个 Web 界面（`/{path}/playground`），无需编写前端代码即可测试 Agent 行为、查看输入输出格式。

### 4. 类型安全

支持 Pydantic 模型作为输入/输出类型，LangServe 会自动生成 JSON Schema 并进行请求验证，确保 API 的健壮性。

### 5. 与 LangGraph 深度集成

LangServe 原生支持 LangGraph 编译后的图（CompiledGraph），包括：
- 状态持久化（Checkpointers）
- 流式输出中间状态
- 多 Agent 编排
- 条件路由和循环图

### 6. 流式输出支持

支持 Server-Sent Events (SSE) 协议，可以实时推送 LLM 生成的 token，适合聊天应用和需要低延迟响应的场景。

---

## 核心功能

### 部署 LangGraph Agent

```python
from fastapi import FastAPI
from langserve import add_routes
from langgraph.graph import StateGraph, MessagesState, START, END

# 构建并编译 LangGraph 图
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

# 创建 FastAPI 应用并注册路由
app = FastAPI(title="LangGraph API")
add_routes(app, graph, path="/chatbot")
```

### 自定义输入类型

使用 Pydantic 模型定义输入格式，LangServe 会自动处理验证和转换：

```python
from pydantic import BaseModel

class ChatInput(BaseModel):
    message: str

add_routes(app, graph, path="/chatbot", input_type=ChatInput)
```

### 输入适配包装

当 LangServe 的输入格式与 LangGraph 期望的状态格式不匹配时，可以使用 `RunnableLambda` 进行适配：

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

def wrap_for_playground(graph):
    def process(input_data):
        message = input_data.get("message", str(input_data))
        return {"messages": [HumanMessage(content=message)]}
    return RunnableLambda(process) | graph

add_routes(app, wrap_for_playground(graph), path="/chatbot", input_type=ChatInput)
```

### 流式输出

```bash
# 使用 curl 测试流式端点
curl -X POST http://localhost:8000/chatbot/stream \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "你好"}]}}'
```

### 批量处理

```bash
# 批量处理多个请求
curl -X POST http://localhost:8000/chatbot/batch \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"messages": [{"role": "user", "content": "天气如何？"}]},
    {"messages": [{"role": "user", "content": "现在几点？"}]}
  ]}'
```

---

## 实战示例：部署 4 种 LangGraph Agent

本项目 `langserve/` 目录下包含 4 个完整的 Agent 示例，均通过 LangServe 部署为 REST API。

### Agent 类型

| 端点 | 描述 | 架构模式 |
|------|------|----------|
| `/chatbot` | 带模拟工具的聊天机器人（天气、时间、计算器） | `StateGraph` + `ToolNode` |
| `/react-agent` | ReAct Agent，带模拟工具（搜索、时间、计算器） | `create_react_agent` |
| `/rag-agent` | RAG Agent，带 LangGraph 知识库检索 | `InMemoryVectorStore` + 检索器 |
| `/multi-agent` | 多 Agent 监督系统，路由到研究员和程序员 | 结构化输出路由 |

### 项目结构

```
langserve/
├── pyproject.toml          # 项目配置与依赖
├── .python-version         # Python 版本指定
├── .env.example            # 环境变量模板
├── server.py               # FastAPI 应用，挂载所有 4 个 Agent
├── chatbot.py              # StateGraph + 模拟工具
├── react_agent.py          # create_react_agent 一行式
├── rag_agent.py            # InMemoryVectorStore + 检索器
├── multi_agent.py          # 监督者 + 研究员/程序员工作者
└── README.md
```

### 快速启动

#### 1. 安装 uv 包管理器

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. 安装依赖

```bash
cd langserve
uv sync
```

#### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，添加 OpenAI API 密钥：

```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

#### 4. 启动服务

```bash
uv run python server.py
```

服务启动后监听 `http://localhost:8000`。

#### 5. 验证服务

- 访问 http://localhost:8000 查看 API 信息
- 访问 http://localhost:8000/docs 查看 OpenAPI 文档
- 访问 Playground 测试界面：
  - http://localhost:8000/chatbot/playground
  - http://localhost:8000/react-agent/playground
  - http://localhost:8000/rag-agent/playground
  - http://localhost:8000/multi-agent/playground

### API 调用示例

```bash
# Chatbot — 查询天气
curl -X POST http://localhost:8000/chatbot/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "东京天气如何？"}]}}'

# ReAct Agent — 网络搜索
curl -X POST http://localhost:8000/react-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "搜索 LangGraph 相关信息"}]}}'

# RAG Agent — 知识库查询
curl -X POST http://localhost:8000/rag-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "StateGraph 是什么？"}]}}'

# Multi-Agent — 需要路由的任务
curl -X POST http://localhost:8000/multi-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "写一个 Python 排序函数"}]}}'
```

---

## 架构详解

### server.py 核心逻辑

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(
    title="LangGraph Agents API",
    version="1.0.0",
    description="4 LangGraph agent examples served via LangServe",
)

# 注册 Agent 路由
add_routes(app, wrap_for_playground(chatbot_graph), path="/chatbot", input_type=ChatInput)
add_routes(app, wrap_for_playground(react_agent_graph), path="/react-agent", input_type=ChatInput)
add_routes(app, wrap_for_playground(rag_agent_graph), path="/rag-agent", input_type=ChatInput)
add_routes(app, wrap_for_playground(multi_agent_graph), path="/multi-agent", input_type=ChatInput)
```

### 输入包装器

由于 LangServe Playground 使用简单文本输入，而 LangGraph 期望 `{"messages": [...]}` 格式的状态，需要编写适配器：

```python
def wrap_for_playground(graph, extra_state=None):
    def process(input_data, config: RunnableConfig = None):
        if isinstance(input_data, dict):
            message = input_data.get("message", str(input_data))
        elif hasattr(input_data, "message"):
            message = input_data.message
        else:
            message = str(input_data)
        
        state = {"messages": [HumanMessage(content=message)]}
        if extra_state:
            state.update(extra_state)
        
        if config is None:
            config = {}
        if "configurable" not in config:
            config["configurable"] = {}
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default"
        return state
    return RunnableLambda(process) | graph
```

---

## 依赖说明

```toml
[project]
dependencies = [
    "langgraph>=0.2.0,<0.4.0",       # LangGraph 框架
    "langchain-openai>=0.3.0,<0.4.0", # OpenAI 集成
    "langchain-core>=0.3.0,<0.4.0",   # LangChain 核心
    "langserve[all]>=0.3.0",          # LangServe 部署框架
    "fastapi>=0.100.0",               # Web 框架
    "uvicorn[standard]>=0.20.0",      # ASGI 服务器
    "python-dotenv>=1.0.1",           # 环境变量加载
    "pydantic>=2.0",                  # 数据验证
]
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `uv: command not found` | 安装 uv（见上方步骤 1） |
| `OPENAI_API_KEY not set` | 确保 `.env` 文件存在且包含 API 密钥 |
| 端口 8000 已被占用 | 在 `server.py` 中修改端口：`uvicorn.run(app, port=8001)` |
| 启动缓慢 | RAG Agent 在启动时构建嵌入向量，需要几秒钟 |

---

## 最佳实践

1. **使用 `uv` 管理依赖**：比传统 `pip` 更快，自动处理虚拟环境
2. **分离 Agent 逻辑与服务层**：Agent 定义在独立文件中，`server.py` 只负责路由注册
3. **使用 Pydantic 模型定义输入**：确保类型安全和自动验证
4. **编写输入适配器**：使用 `RunnableLambda` 桥接 LangServe 和 LangGraph 的格式差异
5. **配置 `thread_id`**：用于会话持久化和多用户隔离
6. **使用结构化输出路由**：多 Agent 场景下，使用 `with_structured_output()` 提高路由准确性

---

## 相关资源

- [LangServe 官方文档](https://python.langchain.com/docs/langserve/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [uv 包管理器](https://docs.astral.sh/uv/)

