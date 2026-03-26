# 导入 A2A 框架的核心模块
from a2a.server.agent_execution import AgentExecutor, RequestContext  # AgentExecutor: Agent 执行器基类; RequestContext: 请求上下文
from a2a.server.events import EventQueue  # EventQueue: 事件队列，用于向客户端发送事件
from a2a.types import (Part, Task, TextPart, UnsupportedOperationError)  # A2A 协议中的数据类型
from a2a.utils import (completed_task, new_artifact)  # 工具函数：创建已完成任务和新的产物(artifact)
from a2a.utils.errors import ServerError  # 服务端错误类


class WeatherAgentExecutor(AgentExecutor):
    """天气 Agent 执行器，继承自 AgentExecutor 基类，负责处理天气查询请求。"""

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """执行天气查询任务。

        Args:
            context: 请求上下文，包含任务 ID、上下文 ID 和用户消息等信息。
            event_queue: 事件队列，用于将执行结果以事件的形式返回给客户端。
        """
        # 模拟天气查询结果（硬编码的示例数据）
        text="""未来 3 天的天气如下：1. 明天（2025年6月1日）：晴天；2. 后天（2025年6月2日）：小雨；3. 大后天（2025年6月3日）：大雨。"""
        # 将查询结果封装为已完成任务事件，并放入事件队列
        event_queue.enqueue_event(
            completed_task(
                context.task_id,       # 任务 ID
                context.context_id,    # 上下文 ID
                [new_artifact(parts=[Part(root=TextPart(text=text))], name="天气查询结果")],  # 创建包含文本结果的产物
                [context.message],     # 原始用户消息
            )
        )

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        """取消任务（当前不支持取消操作，抛出异常）。"""
        raise ServerError(error=UnsupportedOperationError())
