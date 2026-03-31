# LangGraph Chatbot Demo

## Overview

A learning project demonstrating LangGraph patterns through 9 standalone demos, each showcasing a different agent architecture.

## Tech Stack

- Python 3.13 (managed via `uv`, see `pyproject.toml` and `uv.lock`)
- LangGraph 1.1.3, LangChain OpenAI 1.1.12, LangChain Tavily 0.2.17
- LLM: `gpt-4o-mini` (temperature=0)
- Environment variables loaded via `python-dotenv`

## Setup

```bash
uv sync                  # install dependencies
cp .env.example .env     # then fill in OPENAI_API_KEY (required), TAVILY_API_KEY (optional)
```

## Demos

```
chatbot.py                - Basic chatbot: manual StateGraph with nodes, edges, conditional routing
react_agent.py            - Same chatbot using prebuilt create_react_agent helper
human_in_the_loop.py      - interrupt() / Command(resume=...) for human approval of sensitive tools
multi_agent_supervisor.py - Supervisor routes tasks to researcher/coder worker agents
persistence_memory.py     - MemorySaver checkpointer with multi-thread conversation memory
streaming.py              - Real-time token streaming (messages mode) and step updates (updates mode)
reflection.py             - Write → critique → revise loop with custom state and iteration control
plan_and_execute.py       - Planner creates steps, executor runs them, replanner adjusts if needed
rag_agent.py              - InMemoryVectorStore + OpenAIEmbeddings retrieval tool in a ReAct agent
```

## Key Patterns

- Tools are `@tool`-decorated functions from `langchain_core.tools`
- `llm.bind_tools(tools)` attaches tool schemas to the LLM
- `ToolNode(tools)` auto-executes tool calls from AI messages
- Conditional edge `should_use_tools` routes between tool execution and END
- `MessagesState` tracks the conversation message list as graph state
- `interrupt()` + `Command(resume=...)` for human-in-the-loop approval
- `MemorySaver` checkpointer enables persistence and thread management
- `with_structured_output()` constrains LLM to a Pydantic schema for routing
- `app.stream(stream_mode="messages")` for token-level streaming
