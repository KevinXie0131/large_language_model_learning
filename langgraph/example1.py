# https://www.bilibili.com/video/BV1LW4UzLENp
# 7分钟写一个AI Agent代码（基于LangGraph）

from typing import Literal
# pip install langchain-openai
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI  # 只需要导入 ChatOpenAI
# pip install -U langgraph
# pip install langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host files.pythonhosted.org --trusted-host pypi.org --default-timeout=1000
# 导入langgraph 检查点，用于持久化状态
from langgraph.checkpoint.memory import MemorySaver
# 导入状态图和状态
from langgraph.graph import END, StateGraph, MessagesState
# 导入工具节点
from langgraph.prebuilt import ToolNode

url = 'https://api.siliconflow.cn/v1/'
api_key = 'sk-chqxedzvmgomulnfrtyfihscoae zmjlqxvxhnzbtbabqitdb'

# 定义工具函数，用于代理调用外部工具
@tool
def search(query: str):
    """模拟一个搜索工具"""
    if "上海" in query.lower() or "Shanghai" in query.lower():
        return "现在25度，今天有可能下雨。"
    return "现在是35度，阳光明媚。"

# 将工具函数放入工具列表
# 我们做agent有一个最关键的点在于我们会有很多工具，大家可以想想你要做一个旅游规划你要用到哪些工具？
# 所以我们做agent就会存多个工具，agent里面的头脑也就是大模型会去选择使用哪个或哪些工具，然后调用工具，然后返回结果给用户。
tools = [search]

# 创建工具节点，为了让langgraph更好的调用工具，你要将你的工具列表传入工具节点中。这个节点本来什么都没有，但是你把工具给到节点之后这个节点就能调用工具了。
tool_node = ToolNode(tools)

# 1. 初始化模型和工具，定义并绑定工具到模型
# 使用 ChatOpenAI 而不是 OpenAI，因为 bind_tools 是聊天模型的方法
model = ChatOpenAI(
    base_url=url,
    api_key=api_key,
    model="deepseek-ai/DeepSeek-V3"
).bind_tools(tools)

# 定义函数，决定是否继续执行
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # 如果LLM调用了工具，则转到"tools"节点
    if last_message.tool_calls:
        return "tools"
    # 否则，停止（回复用户）
    return END

# 定义调用模型的函数
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # 返回列表，因为这将被添加到现有列表中
    return {"messages": [response]}

# 2. 用状态初始化图，定义一个新的状态图
workflow = StateGraph(MessagesState)

# 3. 定义图节点，定义我们将循环的两个节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 4. 定义入口点和图边
# 设置入口点为"agent"
# 这意味着这是第一个被调用的节点
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    # 首先，定义起始节点。我们使用[agent]。
    # 这意味着这些边是在调用[agent]节点后采取的。
    "agent",
    should_continue,
)

# 添加从[tools]到[agent]的普通边。
# 这意味着在调用[tools]后，接下来调用[agent]节点。
workflow.add_edge("tools", "agent")

# 初始化内存以在图运行之间持久化状态
checkpointer = MemorySaver()

# 5. 编译图
# 这将其编译成一个LangChain可运行对象，
# 这意味着你可以像使用其他可运行对象一样使用它。
# 注意，我们（可选地）在编译图时传递内存
app = workflow.compile(checkpointer=checkpointer)

# 6. 执行图，使用可运行对象
final_state = app.invoke(
    {"messages": [HumanMessage(content="上海的天气怎么样?")]},
    config={"configurable": {"thread_id": 42}}
)

# 从 final_state 中获取最后一条消息的内容
result = final_state["messages"][-1].content
print(result)

# 再次调用，测试记忆功能（使用了相同的 thread_id: 42）
final_state = app.invoke(
    {"messages": [HumanMessage(content="我问的那个城市?")]},
    config={"configurable": {"thread_id": 42}}
)

result = final_state["messages"][-1].content
print(result)

# 使用本地渲染方式生成图表
try:
    # 尝试使用本地渲染方式 (需要安装 pyppeteer 等依赖)
    graph_png = app.get_graph().draw_mermaid_png(draw_method="pyppeteer")
    with open("langgraph_base.png", "wb") as f:
        f.write(graph_png)
except Exception as e:
    print(f"本地渲染失败，使用默认方式：{e}")
    try:
        # 如果本地渲染失败，尝试获取 Mermaid 文本代码
        graph_mermaid = app.get_graph().draw_mermaid()
        print("Mermaid图表代码：")
        print(graph_mermaid)

        # 将mermaid代码保存到文件，可以在支持 Mermaid 的编辑器中查看
        with open("langgraph_base.md", "w", encoding="utf-8") as f:
            f.write(graph_mermaid)
        print("已保存Mermaid代码到 langgraph_base.md 文件")
    except Exception as e2:
        print(f"生成图表失败：{e2}")