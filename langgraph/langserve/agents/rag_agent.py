"""RAG agent with InMemoryVectorStore and retriever tool."""

from datetime import datetime

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

SAMPLE_DOCUMENTS = [
    Document(
        page_content=(
            "LangGraph is a library for building stateful, multi-actor applications "
            "with LLMs. It uses a graph-based approach where nodes represent "
            "computation steps and edges define the flow between them."
        ),
        metadata={"source": "langgraph_overview"},
    ),
    Document(
        page_content=(
            "StateGraph is the main class in LangGraph. You define nodes as Python "
            "functions, add edges between them, and compile the graph into a runnable. "
            "MessagesState is a built-in state type that tracks a list of messages."
        ),
        metadata={"source": "langgraph_core"},
    ),
    Document(
        page_content=(
            "LangGraph supports conditional edges using add_conditional_edges(). "
            "A routing function inspects the current state and returns the name of "
            "the next node to execute, enabling dynamic branching in your graph."
        ),
        metadata={"source": "langgraph_routing"},
    ),
    Document(
        page_content=(
            "The ToolNode in LangGraph automatically executes tool calls from an "
            "LLM response. Combined with bind_tools(), it creates a tool-calling "
            "loop where the agent can use tools and process their results."
        ),
        metadata={"source": "langgraph_tools"},
    ),
    Document(
        page_content=(
            "create_react_agent() is a prebuilt helper that creates a complete "
            "ReAct agent graph in one line. It handles the tool-calling loop, "
            "message management, and routing automatically."
        ),
        metadata={"source": "langgraph_react"},
    ),
    Document(
        page_content=(
            "LangGraph checkpointers enable persistence and memory. MemorySaver "
            "stores state in memory, while SqliteSaver and PostgresSaver provide "
            "durable storage. Use thread_id in config to maintain separate conversations."
        ),
        metadata={"source": "langgraph_persistence"},
    ),
    Document(
        page_content=(
            "LangServe deploys LangChain runnables as REST APIs using FastAPI. "
            "The add_routes() function creates /invoke, /stream, /batch, and "
            "/playground endpoints for any runnable, including compiled LangGraph graphs."
        ),
        metadata={"source": "langserve_overview"},
    ),
    Document(
        page_content=(
            "Multi-agent systems in LangGraph use a supervisor pattern. A supervisor "
            "node routes tasks to specialized worker agents (e.g., researcher, coder). "
            "Each worker reports back, and the supervisor decides the next step."
        ),
        metadata={"source": "langgraph_multi_agent"},
    ),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore.from_documents(SAMPLE_DOCUMENTS, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about LangGraph and LangServe."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[{i}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def get_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about LangGraph and LangServe. "
    "Always search the knowledge base before answering questions about these topics. "
    "Base your answers on the retrieved documents and cite the source when possible."
)

tools = [search_knowledge_base, get_time]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


def chatbot_node(state: MessagesState):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    return {"messages": [llm.invoke(messages)]}


def should_use_tools(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", should_use_tools)
builder.add_edge("tools", "chatbot")

graph = builder.compile()
