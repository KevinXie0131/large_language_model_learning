# 导入类型注解工具
from typing import Any

# 导入环境变量加载工具
from dotenv import load_dotenv
# 导入近似token计数器，用于估算消息长度
from langchain_core.messages.utils import count_tokens_approximately
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入短期记忆中的总结节点，用于自动总结历史对话
from langmem.short_term import SummarizationNode
# 导入内存检查点存储器
from langgraph.checkpoint.memory import InMemorySaver
# 导入预构建的ReAct Agent创建函数
from langgraph.prebuilt import create_react_agent
# 导入Agent的默认状态类
from langgraph.prebuilt.chat_agent_executor import AgentState

# 加载.env文件中的环境变量（如API密钥）
load_dotenv()
# 初始化大语言模型，temperature=0表示输出更稳定、确定性更高
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 使用大模型对历史信息进行总结
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,  # 用于计算消息token数的函数
    model=llm,  # 用于生成摘要的大模型
    max_tokens=256,  # 当消息总token数超过此阈值时触发总结
    max_summary_tokens=128,  # 生成的摘要最大token数
    output_messages_key="llm_input_messages",  # 总结后的消息存储到此状态键中
)

class State(AgentState):
    # 注意：这个状态管理的作用是为了能够保存上一次总结的结果。这样就可以防止每次调用大模型时，都要重新总结历史信息。
    # 这是一个比较常见的优化方式，因为大模型的调用是比较耗时的。
    context: dict[str, Any]

# 创建内存检查点存储器，用于保存Agent状态
checkpointer = InMemorySaver()

# 创建ReAct Agent，绑定总结节点作为模型调用前的钩子
agent = create_react_agent(
    model=llm,  # 使用的大语言模型
    tools=[],  # 无外部工具，纯对话模式
    pre_model_hook=summarization_node,  # 每次调用LLM前先执行总结
    state_schema=State,  # 使用自定义状态类
    checkpointer=checkpointer,  # 检查点存储器，支持多轮对话
)

if __name__ == "__main__":
    # 配置对话线程ID，同一线程内的消息共享上下文
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
        # 获取AI回复内容（消息列表中的最后一条）
        ai_reply = result["messages"][-1].content
        print(f"AI: {ai_reply}")

        # 打印总结后实际发送给大模型的消息
        # 获取总结后实际传给LLM的消息
        llm_input = result.get("llm_input_messages", [])
        if llm_input:
            print(f"  [总结后发送给LLM的消息数量: {len(llm_input)}]")
            for m in llm_input:
                # 获取消息的角色和内容，兼容不同消息格式
                m_role = getattr(m, "type", m.get("role", "unknown") if isinstance(m, dict) else "unknown")
                m_content = getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
                print(f"    [{m_role}] {m_content[:200]}")

        # 查看当前状态中的消息数量和总结内容
        # 获取当前Agent状态快照
        state = agent.get_state(config)
        # 获取状态中保存的全部消息
        all_msgs = state.values.get("messages", [])
        # 获取总结上下文信息
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