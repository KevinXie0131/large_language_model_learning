"""
LangGraph RAG (Retrieval-Augmented Generation) Agent Demo
LangGraph RAG（检索增强生成）代理演示

Shows how to build an agent that retrieves relevant documents from a
vector store before answering questions, combining search with generation.
展示如何构建一个在回答问题前从向量存储中检索相关文档的代理，将搜索与生成相结合。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - Tool-based retrieval: vector search exposed as a LangGraph tool
    基于工具的检索：向量搜索作为 LangGraph 工具暴露
  - InMemoryVectorStore: simple vector store for demo purposes
    InMemoryVectorStore：用于演示的简单内存向量存储
  - OpenAIEmbeddings: converts text to vectors for similarity search
    OpenAIEmbeddings：将文本转换为向量以进行相似度搜索
  - Combining retrieval tool with other tools in a ReAct agent
    在 ReAct 代理中将检索工具与其他工具组合使用

Graph structure (same ReAct loop as chatbot.py):
图结构（与 chatbot.py 相同的 ReAct 循环）：
  START → chatbot → (has tool calls?) → tools → chatbot → ... → END

The key difference: one of the tools performs vector similarity search
over a knowledge base, grounding the LLM's answers in retrieved context.
关键区别：其中一个工具对知识库执行向量相似度搜索，使 LLM 的回答基于检索到的上下文。
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore  # In-memory vector store (for demo) / 内存向量存储（演示用）
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI embedding model / OpenAI 嵌入模型
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Create a sample knowledge base
#    In production, you'd load these from files, databases, or web scraping.
# 1. 创建示例知识库
#    生产环境中，通常从文件、数据库或网页抓取加载数据。
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
# 2. 构建向量存储
# ---------------------------------------------------------------------------

print("Building vector store from sample documents...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Text embedding model: converts text to vectors / 文本嵌入模型：将文本转为向量
vector_store = InMemoryVectorStore.from_documents(  # Create vector store from document list / 从文档列表创建向量存储
    SAMPLE_DOCUMENTS, embedding=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retriever: return top 3 most similar docs / 检索器：返回最相似的3个文档
print(f"Indexed {len(SAMPLE_DOCUMENTS)} documents.\n")

# ---------------------------------------------------------------------------
# 3. Define Tools
# 3. 定义工具
# ---------------------------------------------------------------------------


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information about LangGraph.
    Use this tool when the user asks about LangGraph concepts, features,
    or implementation details.
    搜索知识库获取 LangGraph 相关信息。当用户询问 LangGraph 概念、功能或实现细节时使用。"""
    docs = retriever.invoke(query)  # Execute vector similarity search / 执行向量相似度搜索
    if not docs:
        return "No relevant documents found."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")  # Document source / 文档来源
        topic = doc.metadata.get("topic", "general")  # Document topic / 文档主题
        results.append(f"[{i}] ({source}/{topic}): {doc.page_content}")

    return "\n\n".join(results)  # Join retrieval results and return to LLM / 拼接检索结果返回给 LLM


@tool
def get_current_time() -> str:
    """Get the current date and time. / 获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [search_knowledge_base, get_current_time]

# ---------------------------------------------------------------------------
# 4. LLM + Graph
# 4. LLM + 图
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
    from langchain_core.messages import SystemMessage

    if not messages or not isinstance(messages[0], SystemMessage):  # Ensure system prompt is at the start / 确保系统提示在消息列表开头
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)  # LLM decides: answer directly or search KB first / LLM 决定是直接回答还是先检索知识库
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
# 5. 交互式命令行
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

        # Stream to show retrieval in action / 流式输出，展示检索过程
        print()
        final_content = ""
        for chunk in app.stream(  # Stream graph execution, show retrieval and generation in real-time / 流式执行图，实时显示检索和生成过程
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
