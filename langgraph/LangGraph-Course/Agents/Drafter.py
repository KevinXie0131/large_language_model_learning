# 导入类型提示相关模块
from typing import Annotated, Sequence, TypedDict
# 导入环境变量加载工具
from dotenv import load_dotenv
# 导入各种消息类型
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入工具装饰器
from langchain_core.tools import tool
# 导入消息累加器
from langgraph.graph.message import add_messages
# 导入状态图和终点
from langgraph.graph import StateGraph, END
# 导入预构建的工具节点
from langgraph.prebuilt import ToolNode

# 加载环境变量
load_dotenv()

# This is the global variable to store document content
# 全局变量，用于存储文档内容
document_content = ""

# 定义代理状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 更新文档内容的工具
@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    # 用新内容替换文档内容
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"


# 保存文档到文件的工具
@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """

    global document_content

    # 如果文件名没有.txt后缀，自动添加
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"


    try:
        # 将文档内容写入文件
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."

    except Exception as e:
        return f"Error saving document: {str(e)}"


# 工具列表
tools = [update, save]

# 初始化模型并绑定工具
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# 代理节点：处理用户输入并调用LLM
def our_agent(state: AgentState) -> AgentState:
    # 系统提示词，包含当前文档内容
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.

    The current document content is:{document_content}
    """)

    # 如果没有消息历史，显示欢迎提示
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        # 获取用户输入
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    # 组合所有消息：系统提示 + 历史消息 + 新用户消息
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    # 调用模型
    response = model.invoke(all_messages)

    # 打印AI回复
    print(f"\n🤖 AI: {response.content}")
    # 如果模型决定调用工具，打印工具名称
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    # 返回更新后的消息列表
    return {"messages": list(state["messages"]) + [user_message, response]}


# 条件判断函数：决定是否继续对话
def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]

    # 如果没有消息，继续
    if not messages:
        return "continue"

    # This looks for the most recent tool message....
    # 从最新消息开始往回查找
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        # 检查是否有保存成功的工具消息
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint

    return "continue"

# 打印消息的辅助函数
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return

    # 只打印最近3条消息中的工具结果
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


# 构建状态图
graph = StateGraph(AgentState)

# 添加代理节点和工具节点
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

# 设置入口点
graph.set_entry_point("agent")

# 代理节点执行后总是先进入工具节点
graph.add_edge("agent", "tools")


# 工具执行后通过条件边决定是否继续
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",  # 继续则回到代理节点
        "end": END,  # 结束则退出
    },
)

# 编译图
app = graph.compile()

# Save the graph as a PNG image
# 将图保存为PNG图片
def save_graph_as_png():
    """Save the LangGraph structure as a PNG image."""
    try:
        # Get the graph structure
        # 获取图结构
        graph_obj = app.get_graph()

        # Draw as PNG using mermaid
        # 使用mermaid绘制PNG
        png_data = graph_obj.draw_mermaid_png()

        # Save to file
        # 保存到文件
        output_path = "Drafter_graph.png"
        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"\n📊 Graph saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"\n⚠️ Error saving graph as PNG: {e}")
        # Fallback: save as ASCII art
        # 备选方案：保存为ASCII图
        try:
            ascii_graph = graph_obj.draw_ascii()
            print("\n📊 Graph structure (ASCII):")
            print(ascii_graph)
        except:
            pass
        return None

# Save the graph when the script is imported
# 脚本加载时保存图
save_graph_as_png()

# 运行文档代理的主函数
def run_document_agent():
    print("\n ===== DRAFTER =====")

    # 初始化空状态
    state = {"messages": []}

    # 以流式模式运行代理
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")

# 程序入口点
if __name__ == "__main__":
    run_document_agent()
