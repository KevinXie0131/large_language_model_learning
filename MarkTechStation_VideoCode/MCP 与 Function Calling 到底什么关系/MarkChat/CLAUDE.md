# MarkChat

A Flask web app demonstrating the relationship between MCP (Model Context Protocol) and Function Calling.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenRouter API key

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create a `.env` file in the project root with your API key:

```
OPENROUTER_API_KEY=your_key_here
```

## Run

Start the Flask web app:

```bash
uv run start.py
```

Then open http://127.0.0.1:5000/ in your browser.

## Test MCP Client independently

```bash
uv run mcp_client.py
```

## Project Structure

- `start.py` — Flask app entry point (routes and API)
- `backend.py` — LLM processing, Function Calling logic, and MCP integration
- `mcp_server.py` — MCP server exposing tools via stdio transport
- `mcp_client.py` — MCP client connecting to the server
- `templates/` — HTML templates
- `static/` — CSS and JavaScript
