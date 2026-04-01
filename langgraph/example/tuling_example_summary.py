from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any

# 使用大模型对历史信息进行总结
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=llm,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

class State(AgentState):
    # 注意：这个状态管理的作用是为了能够保存上一次总结的结果。这样就可以防止每次调用大模型时，都要重新总结历史信息。
    # 这是一个比较常见的优化方式，因为大模型的调用是比较耗时的。
    context: dict[str, Any]

checkpointer = InMemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[],
    pre_model_hook=summarization_node,
    state_schema=State,
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
        print(f"AI: {ai_reply[:200]}")

        # 查看当前状态中的消息数量和总结内容
        state = agent.get_state(config)
        all_msgs = state.values.get("messages", [])
        context = state.values.get("context", {})
        print(f"  [状态中的消息数量: {len(all_msgs)}]")

        # 打印总结内容（context 是 SummarizationNode 存储摘要的地方）
        if context:
            print(f"  [context 内容]: {context}")

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