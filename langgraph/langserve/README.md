# LangServe + LangGraph Examples

4 LangGraph agent examples served as REST APIs via LangServe.

## Agents

| Endpoint | Description | Pattern |
|----------|-------------|---------|
| `/chatbot` | Chatbot with mock tools (weather, time, calculator) | `StateGraph` + `ToolNode` |
| `/react-agent` | ReAct agent with mock tools (search, time, calculator) | `create_react_agent` |
| `/rag-agent` | RAG agent with LangGraph knowledge base | `InMemoryVectorStore` + retriever |
| `/multi-agent` | Supervisor routing to researcher & coder workers | Structured output routing |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- OpenAI API key

## Build & Run Locally

### 1. Install uv (if not installed)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
cd langserve
uv sync
```

This creates a `.venv` virtual environment and installs all packages from `pyproject.toml`.

### 3. Configure environment variables

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 4. Run the server

```bash
uv run python server.py
```

The server starts at http://localhost:8000.

### 5. Verify it's working

Open http://localhost:8000 in your browser — you should see a JSON response listing all available endpoints.

Open http://localhost:8000/docs for the interactive OpenAPI documentation.

## Project Structure

```
langserve/
├── pyproject.toml          # Project config & dependencies
├── .python-version         # Python 3.12
├── .env.example            # Environment variable template
├── server.py               # FastAPI app, mounts all 4 agents
├── README.md
└── agents/
    ├── __init__.py          # Re-exports all graphs
    ├── chatbot.py           # StateGraph + mock tools
    ├── react_agent.py       # create_react_agent one-liner
    ├── rag_agent.py         # InMemoryVectorStore + retriever
    └── multi_agent.py       # Supervisor + researcher/coder workers
```

## Endpoints

Each agent exposes these endpoints:

- `POST /{agent}/invoke` — Single invocation
- `POST /{agent}/batch` — Batch processing
- `POST /{agent}/stream` — Streaming (SSE)
- `GET /{agent}/playground` — Interactive playground UI
- `GET /docs` — OpenAPI documentation

## Example Usage

```bash
# Chatbot — ask about weather
curl -X POST http://localhost:8000/chatbot/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}}'

# ReAct Agent — web search
curl -X POST http://localhost:8000/react-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Search for info about LangGraph"}]}}'

# RAG Agent — knowledge base query
curl -X POST http://localhost:8000/rag-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "What is StateGraph in LangGraph?"}]}}'

# Multi-Agent — task requiring routing
curl -X POST http://localhost:8000/multi-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Write a Python function to sort a list"}]}}'
```

## Playground

Visit the interactive playground UI for each agent:

- http://localhost:8000/chatbot/playground
- http://localhost:8000/react-agent/playground
- http://localhost:8000/rag-agent/playground
- http://localhost:8000/multi-agent/playground

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv: command not found` | Install uv (see step 1 above) |
| `OPENAI_API_KEY not set` | Make sure `.env` exists with your API key |
| Port 8000 already in use | Change the port in `server.py`: `uvicorn.run(app, port=8001)` |
| Slow startup | The RAG agent builds embeddings at startup — this takes a few seconds |
