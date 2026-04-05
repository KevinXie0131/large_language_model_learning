# 导入环境变量加载工具
from dotenv import load_dotenv
# 导入操作系统模块
import os
# 导入状态图和终点
from langgraph.graph import StateGraph, END
# 导入类型提示相关模块
from typing import TypedDict, Annotated, Sequence
# 导入各种消息类型
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
# 导入消息累加器（用作状态的reducer）
from operator import add as add_messages
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入OpenAI嵌入模型
from langchain_openai import OpenAIEmbeddings
# 导入PDF文档加载器
from langchain_community.document_loaders import PyPDFLoader
# 导入递归字符文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入Chroma向量数据库
from langchain_chroma import Chroma
# 导入工具装饰器
from langchain_core.tools import tool

# 加载环境变量
load_dotenv()

# 初始化GPT-4o模型，temperature=0减少幻觉
llm = ChatOpenAI(
    model="gpt-4o", temperature = 0) # I want to minimize hallucination - temperature = 0 makes the model output more deterministic

# Our Embedding Model - has to also be compatible with the LLM
# 初始化嵌入模型，用于将文本转换为向量
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)


# PDF文件路径
pdf_path = "Stock_Market_Performance_2024.pdf"


# Safety measure I have put for debugging purposes :)
# 检查PDF文件是否存在
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# 加载PDF文件
pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

# Checks if the PDF is there
# 尝试加载PDF并打印页数
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
# 文本分块：将长文档切分为小块，便于检索
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # 每块最大1000个字符
    chunk_overlap=200  # 相邻块重叠200个字符，保持上下文连贯
)


# 对PDF页面执行分块操作
pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

# Use current script directory for persistence
# 使用当前脚本目录作为持久化存储路径
persist_directory = os.path.dirname(os.path.abspath(__file__))
# 集合名称
collection_name = "stock_market"

# If our collection does not exist in the directory, we create using the os command
# 如果目录不存在则创建
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here, we actually create the chroma database using our embeddigns model
    # 使用嵌入模型创建Chroma向量数据库
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever
# 创建检索器，使用相似度搜索
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

# 定义检索工具：从向量数据库中搜索相关文档
@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """

    # 执行检索查询
    docs = retriever.invoke(query)

    # 如果没有找到相关文档
    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."

    # 格式化检索结果
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


# 工具列表
tools = [retriever_tool]

# 将工具绑定到LLM
llm = llm.bind_tools(tools)

# 定义代理状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 条件判断函数：检查是否需要调用工具
def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    # 如果最后一条消息包含工具调用，返回True
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


# 系统提示词：定义AI助手的角色和行为
system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""


# 创建工具名称到工具对象的映射字典
tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
# LLM代理节点：调用大语言模型
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    # 在消息前加上系统提示
    messages = [SystemMessage(content=system_prompt)] + messages
    # 调用LLM
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent
# 检索代理节点：执行工具调用
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    # 获取LLM返回的工具调用列表
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict: # Checks if a valid tool is present
            # 工具名称无效时返回错误提示
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            # 执行工具调用
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")


        # Appends the Tool Message
        # 将工具执行结果封装为ToolMessage
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


# 构建状态图
graph = StateGraph(AgentState)
# 添加LLM节点
graph.add_node("llm", call_llm)
# 添加检索代理节点
graph.add_node("retriever_agent", take_action)

# 添加条件边：根据是否有工具调用决定下一步
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}  # True则执行检索，False则结束
)
# 检索完成后返回LLM节点
graph.add_edge("retriever_agent", "llm")
# 设置入口点为LLM节点
graph.set_entry_point("llm")

# 编译RAG代理
rag_agent = graph.compile()


# 运行RAG代理的主函数
def running_agent():
    print("\n=== RAG AGENT===")

    # 循环接受用户提问
    while True:
        user_input = input("\nWhat is your question: ")
        # 输入exit或quit退出
        if user_input.lower() in ['exit', 'quit']:
            break

        # 将用户输入转换为HumanMessage类型
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        # 调用RAG代理处理问题
        result = rag_agent.invoke({"messages": messages})

        # 打印最终回答
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


# 启动代理
running_agent()
