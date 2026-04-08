"""
LangGraph Cross-Thread Memory Demo (using InMemoryStore)
LangGraph 跨线程记忆演示（使用 InMemoryStore）

Shows how to share memory across different conversation threads using
LangGraph's Store API. The agent remembers user facts across separate
conversations, simulating long-term user memory.
展示如何使用 LangGraph 的 Store API 在不同对话线程间共享记忆。
代理在不同对话中记住用户信息，模拟长期用户记忆。

Key LangGraph concepts demonstrated:
演示的 LangGraph 关键概念：
  - InMemoryStore: key-value store that persists across threads
    InMemoryStore：跨线程持久化的键值存储
  - store.put() / store.search(): save and retrieve memories
    store.put() / store.search()：保存和检索记忆
  - Combining MemorySaver (thread state) with Store (cross-thread data)
    结合 MemorySaver（线程状态）和 Store（跨线程数据）
  - Memory injection via node logic
    通过节点逻辑注入记忆

Graph structure:
图结构：
  START → recall_memories → chatbot → save_memories → END
"""

import uuid
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Per-thread state / 每线程状态
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore  # Cross-thread memory / 跨线程记忆

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 1. Set up the Store
#    InMemoryStore provides a namespace-based key-value store.
#    Each user gets their own namespace for storing facts.
# 1. 设置存储
#    InMemoryStore 提供基于命名空间的键值存储。
#    每个用户有自己的命名空间来存储事实。
# ---------------------------------------------------------------------------

store = InMemoryStore()  # Cross-thread memory store / 跨线程记忆存储
memory = MemorySaver()   # Per-thread checkpointer / 每线程检查点保存器

# ---------------------------------------------------------------------------
# 2. Define Nodes
# 2. 定义节点
# ---------------------------------------------------------------------------


def recall_memories_node(state: MessagesState, config: dict, *, store: InMemoryStore) -> dict:
    """Load existing memories about the user and inject them as context.
    加载关于用户的现有记忆并将其注入为上下文。

    The store parameter is automatically injected by LangGraph when
    the graph is compiled with a store.
    当图与 store 一起编译时，LangGraph 会自动注入 store 参数。
    """
    user_id = config["configurable"].get("user_id", "default")
    namespace = ("user_memories", user_id)

    # Search for all memories in this user's namespace / 在用户命名空间中搜索所有记忆
    memories = store.search(namespace)
    memory_texts = [m.value["fact"] for m in memories] if memories else []

    if memory_texts:
        memory_context = "Known facts about this user:\n" + "\n".join(f"- {m}" for m in memory_texts)
    else:
        memory_context = "No prior memories about this user."

    # Inject memories as a system message at the start of the conversation
    # 将记忆作为系统消息注入到对话开头
    system_msg = SystemMessage(content=(
        f"You are a helpful assistant with long-term memory. "
        f"You remember facts about users across conversations.\n\n"
        f"{memory_context}\n\n"
        f"If the user shares personal facts (name, preferences, etc.), "
        f"acknowledge them naturally. You'll remember them for next time."
    ))

    return {"messages": [system_msg] + state["messages"]}


def chatbot_node(state: MessagesState) -> dict:
    """Call the LLM with the current message history (including injected memories).
    使用当前消息历史（包括注入的记忆）调用 LLM。"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def save_memories_node(state: MessagesState, config: dict, *, store: InMemoryStore) -> dict:
    """Extract and save any new facts the user shared in this conversation.
    提取并保存用户在此对话中分享的新事实。

    Uses the LLM to identify factual statements worth remembering.
    使用 LLM 识别值得记住的事实陈述。
    """
    user_id = config["configurable"].get("user_id", "default")
    namespace = ("user_memories", user_id)

    # Get the last user message / 获取最后一条用户消息
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        return {}

    last_user_msg = user_messages[-1].content

    # Ask LLM to extract facts / 让 LLM 提取事实
    extraction = llm.invoke([
        SystemMessage(content=(
            "Extract personal facts from the user message that are worth remembering. "
            "Return ONLY a comma-separated list of facts, or 'NONE' if no memorable facts.\n"
            "Examples of memorable facts: name, age, location, preferences, job, hobbies.\n"
            "Examples of NOT memorable: greetings, questions, generic statements."
        )),
        HumanMessage(content=last_user_msg)
    ])

    if extraction.content.strip().upper() != "NONE":
        facts = [f.strip() for f in extraction.content.split(",") if f.strip()]
        for fact in facts:
            # Store each fact with a unique key / 用唯一键存储每个事实
            store.put(namespace, str(uuid.uuid4()), {"fact": fact})
            print(f"  💾 Saved memory: {fact}")

    return {}


# ---------------------------------------------------------------------------
# 3. Build the Graph
# 3. 构建图
# ---------------------------------------------------------------------------

graph = StateGraph(MessagesState)
graph.add_node("recall_memories", recall_memories_node)
graph.add_node("chatbot", chatbot_node)
graph.add_node("save_memories", save_memories_node)

graph.add_edge(START, "recall_memories")    # First: load memories / 首先：加载记忆
graph.add_edge("recall_memories", "chatbot")  # Then: chat with context / 然后：带上下文聊天
graph.add_edge("chatbot", "save_memories")    # Finally: save new facts / 最后：保存新事实
graph.add_edge("save_memories", END)

app = graph.compile(checkpointer=memory, store=store)

# ---------------------------------------------------------------------------
# 4. Interactive CLI Loop
#    Shows cross-thread memory by using different thread_ids but same user_id.
# 4. 交互式命令行循环
#    通过使用不同的 thread_id 但相同的 user_id 来展示跨线程记忆。
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  LangGraph Cross-Thread Memory Demo")
    print("  LangGraph 跨线程记忆演示")
    print("  Memory persists across conversations!")
    print("  Commands: 'new' = start new thread, 'quit' = exit")
    print("=" * 60)

    user_id = "demo_user"
    thread_id = 1
    messages = []

    print(f"\n[Thread {thread_id}] New conversation started")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "new":
            thread_id += 1
            messages = []
            print(f"\n[Thread {thread_id}] New conversation started (memories carry over!)")
            continue

        messages.append(HumanMessage(content=user_input))

        config = {
            "configurable": {
                "thread_id": str(thread_id),
                "user_id": user_id,
            }
        }
        result = app.invoke({"messages": messages}, config=config)

        ai_message = result["messages"][-1]
        print(f"\nAssistant: {ai_message.content}")
        messages = result["messages"]


if __name__ == "__main__":
    main()
