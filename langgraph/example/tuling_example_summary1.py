"""
LangGraph 消息裁剪示例：基于 trim_messages 的上下文窗口管理

演示如何使用 pre_model_hook + trim_messages 在调用大模型前裁剪历史消息，
只保留最近的消息（按token数限制），避免超出模型的上下文窗口限制。
与 tuling_example_summary.py 的区别：这里是直接截断旧消息，而不是生成摘要。
"""

# 导入环境变量加载工具
from dotenv import load_dotenv
# 导入消息裁剪函数和近似token计数器
from langchain_core.messages.utils import (
    trim_messages,  # 按策略裁剪消息列表
    count_tokens_approximately,  # 近似计算消息的token数
)
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入内存检查点存储器
from langgraph.checkpoint.memory import InMemorySaver
# 导入预构建的ReAct Agent创建函数
from langgraph.prebuilt import create_react_agent

# 加载.env文件中的环境变量
load_dotenv()
# 初始化大语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 模型调用前的钩子函数：每次调用LLM前自动裁剪历史消息
def pre_model_hook(state):
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",       # 保留最新的消息（从末尾开始）
        token_counter=count_tokens_approximately,  # 使用近似token计数
        max_tokens=384,        # 最大保留384个token
        start_on="human",      # 裁剪后的消息必须以human消息开头
        end_on=("human", "tool"),  # 裁剪后的消息必须以human或tool消息结尾
    )
    # 返回裁剪后的消息列表，作为实际发送给LLM的输入
    return {"llm_input_messages": trimmed_messages}


# 创建内存检查点存储器
checkpointer = InMemorySaver()

# 创建Agent，绑定 pre_model_hook 实现消息裁剪
agent = create_react_agent(
    model=llm,
    tools=[],                        # 无工具，纯对话Agent
    pre_model_hook=pre_model_hook,   # 每次调用LLM前执行裁剪
    checkpointer=checkpointer,
)

if __name__ == "__main__":
    # 配置对话线程ID
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
        # 调用Agent处理用户消息
        result = agent.invoke({"messages": [{"role": "user", "content": msg}]}, config)
        # 获取AI回复内容
        ai_reply = result["messages"][-1].content
        print(f"AI: {ai_reply}")

        # 查看当前状态中的消息数量
        state = agent.get_state(config)
        # 获取状态中保存的全部消息列表
        all_msgs = state.values.get("messages", [])

        # 打印裁剪后实际发送给大模型的消息
        # 获取裁剪后实际传给LLM的消息
        llm_input = state.values.get("llm_input_messages", [])
        if llm_input:
            print(f"  [裁剪后发送给LLM的消息数量: {len(llm_input)}]")
            for m in llm_input:
                # 获取消息的角色和内容，兼容不同消息格式
                m_role = getattr(m, "type", m.get("role", "unknown") if isinstance(m, dict) else "unknown")
                m_content = getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
                # 截取前200个字符作为预览
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

    # 获取最终状态，查看裁剪后实际发送给LLM的消息
    final_state = agent.get_state(config)
    llm_input = final_state.values.get("llm_input_messages", [])
    if llm_input:
        print(f"\n  [裁剪后发送给LLM的消息数量: {len(llm_input)}]")
        for m in llm_input:
            m_role = getattr(m, "type", m.get("role", "unknown") if isinstance(m, dict) else "unknown")
            m_content = getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
            print(f"    [{m_role}] {m_content[:200]}")

    # 打印最终状态，查看总结效果
    # 获取最终的Agent状态
    state = agent.get_state(config)
    print("\n" + "=" * 60)
    print("最终状态中的所有消息：")
    print("=" * 60)
    for msg in state.values.get("messages", []):
        # 提取消息角色和内容，截取前100个字符作为预览
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        preview = content[:100] if content else "(empty)"
        print(f"  [{role}] {preview}")
