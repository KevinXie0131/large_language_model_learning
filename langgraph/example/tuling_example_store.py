from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 定义长期存储
store = InMemoryStore()
# 添加一些测试数据。users是命名空间，user_123是key，后面的JSON数据是value
store.put(
    ("users",),
    "user_123",
    {
        "name": "楼兰",
        "age": "33",
    },
)


# 定义工具
@tool(return_direct=True)
def get_user_info(config: RunnableConfig) -> str:
    """查找用户信息"""
    # 获取长期存储。获取到了后，这个存储组件可读也可写
    store = get_store()
    # 获取配置中的用户ID
    user_id = config["configurable"].get("user_id")
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


agent = create_react_agent(
    model=llm,
    tools=[get_user_info],
    store=store,
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "查找用户信息"}]},
        config={"configurable": {"user_id": "user_123"}},
    )
    print(result["messages"][-1].content)
