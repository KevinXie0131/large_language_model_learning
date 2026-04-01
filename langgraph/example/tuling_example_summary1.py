from dotenv import load_dotenv
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# This function will be called every time before the node that calls LLM
def pre_model_hook(state):
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}


checkpointer = InMemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[],
    pre_model_hook=pre_model_hook,
    checkpointer=checkpointer,
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-thread"}}

    # 第一轮：发送多条消息，积累足够的历史让总结生效
    conversations = [
        "你好，我叫小明，我是一个Python开发者。",
        "我最近在学习LangChain和LangGraph框架。",
        "我对大语言模型的应用非常感兴趣，特别是Agent方向。",
        "我之前做过一个基于RAG的知识库问答系统。",
        "我还用过LangChain的Tool和Memory功能。",
        "我目前在研究如何优化大模型的上下文管理。",
    ]

    print("=" * 60)
    print("开始多轮对话，观察总结效果")
    print("=" * 60)

    for i, msg in enumerate(conversations, 1):
        print(f"\n--- 第 {i} 轮对话 ---")
        print(f"用户: {msg}")
        result = agent.invoke({"messages": [{"role": "user", "content": msg}]}, config)
        ai_reply = result["messages"][-1].content
        print(f"AI: {ai_reply}")

        # 查看当前状态中的消息数量
        state = agent.get_state(config)
        all_msgs = state.values.get("messages", [])

        # 打印裁剪后实际发送给大模型的消息
        llm_input = state.values.get("llm_input_messages", [])
        if llm_input:
            print(f"  [裁剪后发送给LLM的消息数量: {len(llm_input)}]")
            for m in llm_input:
                m_role = getattr(m, "type", m.get("role", "unknown") if isinstance(m, dict) else "unknown")
                m_content = getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
                print(f"    [{m_role}] {m_content[:200]}")
        print(f"  [状态中的消息数量: {len(all_msgs)}]")

    # 最后一轮：测试AI是否记住了之前的信息
    print("\n" + "=" * 60)
    print("测试：AI是否还记得之前的对话内容？")
    print("=" * 60)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请问我叫什么名字？我之前做过什么项目？"}]},
        config,
    )
    print(f"\n用户: 请问我叫什么名字？我之前做过什么项目？")
    print(f"AI: {result['messages'][-1].content}")

    # 打印裁剪后实际发送给LLM的消息
    final_state = agent.get_state(config)
    llm_input = final_state.values.get("llm_input_messages", [])
    if llm_input:
        print(f"\n  [裁剪后发送给LLM的消息数量: {len(llm_input)}]")
        for m in llm_input:
            m_role = getattr(m, "type", m.get("role", "unknown") if isinstance(m, dict) else "unknown")
            m_content = getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
            print(f"    [{m_role}] {m_content[:200]}")

    # 打印最终状态，查看总结效果
    state = agent.get_state(config)
    print("\n" + "=" * 60)
    print("最终状态中的所有消息：")
    print("=" * 60)
    for msg in state.values.get("messages", []):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        preview = content[:100] if content else "(empty)"
        print(f"  [{role}] {preview}")
