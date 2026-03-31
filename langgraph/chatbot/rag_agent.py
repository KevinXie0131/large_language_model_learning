"""
LangGraph RAG (Retrieval-Augmented Generation) Agent Demo

Shows how to build an agent that retrieves relevant documents from a
vector store before answering questions, combining search with generation.

Key LangGraph concepts demonstrated:
  - Tool-based retrieval: vector search exposed as a LangGraph tool
  - InMemoryVectorStore: simple vector store for demo purposes
  - OpenAIEmbeddings: converts text to vectors for similarity search
  - Combining retrieval tool with other tools in a ReAct agent

Graph structure (same ReAct loop as chatbot.py):
  START → chatbot → (has tool calls?) → tools → chatbot → ... → END

The key difference: one of the tools performs vector similarity search
over a knowledge base, grounding the LLM's answers in retrieved context.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Create a sample knowledge base
#    In production, you'd load these from files, databases, or web scraping.
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    Document(
        page_content=(
            "LangGraph is a framework for building stateful, multi-actor "
            "applications with LLMs. It extends LangChain with graph-based "
            "orchestration, enabling cycles, branching, and persistence."
        ),
        metadata={"source": "langgraph-docs", "topic": "overview"},
    ),
    Document(
        page_content=(
            "StateGraph is the core class in LangGraph. You define nodes "
            "(functions that process state), edges (connections between nodes), "
            "and conditional edges (routing logic). The state is a TypedDict "
            "that flows through the graph."
        ),
        metadata={"source": "langgraph-docs", "topic": "stategraph"},
    ),
    Document(
        page_content=(
            "LangGraph supports persistence via checkpointers. MemorySaver "
            "stores state in RAM, while SqliteSaver and PostgresSaver provide "
            "durable storage. Checkpointing enables features like "
            "human-in-the-loop, time-travel, and conversation memory."
        ),
        metadata={"source": "langgraph-docs", "topic": "persistence"},
    ),
    Document(
        page_content=(
            "The ReAct (Reasoning + Acting) pattern is a common agent design. "
            "The agent reasons about what to do, takes an action (tool call), "
            "observes the result, and repeats until the task is complete. "
            "LangGraph's create_react_agent() implements this pattern."
        ),
        metadata={"source": "langgraph-docs", "topic": "react-pattern"},
    ),
    Document(
        page_content=(
            "Tools in LangGraph are Python functions decorated with @tool. "
            "The decorator extracts the function's name, docstring, and type "
            "hints to create a schema the LLM can use for structured tool calls. "
            "ToolNode automatically executes tool calls from AI messages."
        ),
        metadata={"source": "langgraph-docs", "topic": "tools"},
    ),
    Document(
        page_content=(
            "LangGraph supports streaming via stream() and astream(). "
            "stream_mode='updates' yields node-level state changes, while "
            "stream_mode='messages' yields individual LLM tokens for real-time "
            "chat experiences."
        ),
        metadata={"source": "langgraph-docs", "topic": "streaming"},
    ),
    Document(
        page_content=(
            "Multi-agent systems in LangGraph use a supervisor pattern. "
            "A supervisor node decides which worker agent should handle the "
            "next step. Workers can be specialized agents with different tools "
            "and system prompts. Results flow back to the supervisor."
        ),
        metadata={"source": "langgraph-docs", "topic": "multi-agent"},
    ),
    Document(
        page_content=(
            "Human-in-the-loop is implemented using interrupt() and "
            "Command(resume=...). When a node calls interrupt(), the graph "
            "pauses and saves state. The caller can inspect the interrupt "
            "value, get user input, and resume execution."
        ),
        metadata={"source": "langgraph-docs", "topic": "human-in-the-loop"},
    ),
]

# ---------------------------------------------------------------------------
# 2. Build the vector store
# ---------------------------------------------------------------------------

print("Building vector store from sample documents...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore.from_documents(
    SAMPLE_DOCUMENTS, embedding=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print(f"Indexed {len(SAMPLE_DOCUMENTS)} documents.\n")

# ---------------------------------------------------------------------------
# 3. Define Tools
# ---------------------------------------------------------------------------


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information about LangGraph.
    Use this tool when the user asks about LangGraph concepts, features,
    or implementation details."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        topic = doc.metadata.get("topic", "general")
        results.append(f"[{i}] ({source}/{topic}): {doc.page_content}")

    return "\n\n".join(results)


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [search_knowledge_base, get_current_time]

# ---------------------------------------------------------------------------
# 4. LLM + Graph
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base
about LangGraph. When answering questions about LangGraph, ALWAYS search the
knowledge base first to ground your answers in the retrieved documents.

Cite the source when using information from the knowledge base.
If the knowledge base doesn't have relevant info, say so and answer
from your general knowledge."""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def chatbot_node(state: MessagesState):
    messages = state["messages"]
    # Prepend system prompt if not already there
    from langchain_core.messages import SystemMessage

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_use_tools(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(tools)

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", should_use_tools)
graph.add_edge("tools", "chatbot")

app = graph.compile()

# ---------------------------------------------------------------------------
# 5. Interactive CLI
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph RAG Agent Demo")
    print("  Knowledge base: LangGraph documentation (8 documents)")
    print("  The agent searches the KB before answering questions")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print("\nTry: 'What is StateGraph and how does it work?'")
    print("Or:  'How does human-in-the-loop work in LangGraph?'")
    print("Or:  'What streaming modes does LangGraph support?'\n")

    messages = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Stream to show retrieval in action
        print()
        final_content = ""
        for chunk in app.stream(
            {"messages": messages}, stream_mode="updates"
        ):
            for node_name, update in chunk.items():
                if node_name == "tools":
                    tool_msgs = update.get("messages", [])
                    for msg in tool_msgs:
                        if hasattr(msg, "name") and msg.name == "search_knowledge_base":
                            print(f"[Retrieved from knowledge base]")
                            # Show a brief preview of results
                            lines = msg.content.split("\n")
                            for line in lines[:3]:
                                if line.strip():
                                    print(f"  {line[:100]}...")
                            print()
                elif node_name == "chatbot":
                    msgs = update.get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "content") and msg.content:
                            final_content = msg.content

        if final_content:
            print(f"Assistant: {final_content}\n")
            messages.append({"role": "assistant", "content": final_content})


if __name__ == "__main__":
    main()
