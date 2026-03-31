# LangGraph Chatbot with Tools Demo

A demo project showing how to build tool-calling chatbots and agents using [LangGraph](https://langchain-ai.github.io/langgraph/).

## What's Included

| File | Description |
|------|-------------|
| `chatbot.py` | **Chatbot** - builds the agent from scratch using `StateGraph`, nodes, edges, and conditional routing |
| `react_agent.py` | **ReAct Agent** - same functionality using the prebuilt `create_react_agent` helper |
| `human_in_the_loop.py` | **Human-in-the-Loop** - pauses for user approval before executing sensitive tools using `interrupt()` |
| `multi_agent_supervisor.py` | **Multi-Agent Supervisor** - supervisor routes tasks to specialized worker agents (researcher, coder) |
| `persistence_memory.py` | **Persistence & Memory** - conversation state persists across turns with multiple threads via checkpointing |
| `streaming.py` | **Streaming** - real-time token streaming and graph step updates |
| `reflection.py` | **Reflection / Self-Correction** - write → critique → revise loop until quality threshold is met |
| `plan_and_execute.py` | **Plan-and-Execute** - creates a step-by-step plan, executes each step, re-plans if needed |
| `rag_agent.py` | **RAG Agent** - retrieves documents from a vector store before answering questions |

chatbot.py                │ 工具装饰器、LLM初始化、工具绑定、状态图构建、条件路由、消息历史管理   
react_agent.py            │ 预构建代理说明、一行创建等价关系、对话历史更新   
human_in_the_loop.py      │ interrupt/Command机制、工具风险分类、审批流程、检查点必要性、中断恢复循环
multi_agent_supervisor.py │ 结构化输出、路由决定、工作节点专属提示、自定义状态(add_messages注解)、主管-工人循环
persistence_memory.py     │ MemorySaver检查点、线程管理、状态持久化、get_state读取   
streaming.py              │ AIMessageChunk类型、updates模式(节点级)、messages模式(token级)、flush实时显示      
reflection.py             │ 自定义状态字段、迭代控制、评级解析、写作-评论循环边  
plan_and_execute.py       │ 计划状态字段、步骤解析、执行上下文构建、重新规划判断、三节点流程
rag_agent.py              │ 向量存储/嵌入模型、相似度检索、系统提示注入、检索+生成流程    

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
   - `OPENAI_API_KEY` (required)
   - `TAVILY_API_KEY` (optional, for web search)

## Usage

Try these prompts:
- "What time is it?"
- "What is 25 * 17 + 3?"
- "Search the web for the latest news about AI"

```bash
python chatbot.py                # Basic chatbot with StateGraph
python react_agent.py            # Prebuilt create_react_agent
python human_in_the_loop.py      # Approval workflow for sensitive tools
python multi_agent_supervisor.py # Supervisor + worker agents
python persistence_memory.py     # Multi-thread conversation memory
python streaming.py              # Real-time token/step streaming
python reflection.py             # Write-critique-revise loop
python plan_and_execute.py       # Plan steps then execute them
python rag_agent.py              # Retrieval-augmented generation
```

## Key LangGraph Concepts

- **StateGraph** - the core graph class that manages state transitions
- **MessagesState** - built-in state schema that tracks a list of chat messages
- **Nodes** - functions that process state and return updates
- **Edges** - connections between nodes (static or conditional)
- **ToolNode** - prebuilt node that executes tool calls from the LLM
- **create_react_agent** - high-level helper that builds the entire ReAct loop in one call
- **interrupt() / Command(resume=...)** - pause and resume graph execution for human approval
- **MemorySaver** - in-memory checkpointer for state persistence across turns
- **stream() / stream_mode** - real-time streaming of tokens or graph updates
- **with_structured_output()** - constrain LLM output to a Pydantic schema (used for routing)
- **InMemoryVectorStore** - simple vector store for RAG retrieval
