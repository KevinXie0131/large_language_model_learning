"""LangServe server exposing 4 LangGraph agent examples as REST APIs."""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from langserve import add_routes  # noqa: E402

from agents import (  # noqa: E402
    chatbot_graph,
    react_agent_graph,
    rag_agent_graph,
    multi_agent_graph,
)

app = FastAPI(
    title="LangGraph Agents API",
    version="1.0.0",
    description="4 LangGraph agent examples served via LangServe",
)


@app.get("/")
async def root():
    return {
        "message": "LangGraph Agents API",
        "endpoints": {
            "/chatbot": "Chatbot with mock tools (weather, time, calculator)",
            "/react-agent": "ReAct agent with mock tools (search, time, calculator)",
            "/rag-agent": "RAG agent with knowledge base about LangGraph",
            "/multi-agent": "Multi-agent supervisor with researcher and coder",
        },
        "playground": {
            "/chatbot/playground": "Chatbot playground",
            "/react-agent/playground": "ReAct agent playground",
            "/rag-agent/playground": "RAG agent playground",
            "/multi-agent/playground": "Multi-agent playground",
        },
        "docs": "/docs",
    }


add_routes(app, chatbot_graph, path="/chatbot")
add_routes(app, react_agent_graph, path="/react-agent")
add_routes(app, rag_agent_graph, path="/rag-agent")
add_routes(app, multi_agent_graph, path="/multi-agent")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
