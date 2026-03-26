from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (Part, Task, TextPart, UnsupportedOperationError)
from a2a.utils import (completed_task, new_artifact)
from a2a.utils.errors import ServerError


# 天气 Agent 的执行器，继承自 AgentExecutor，负责处理天气查询请求
class WeatherAgentExecutor(AgentExecutor):

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # 模拟天气查询结果（非流式，一次性返回完整结果）
        text="""您要查询的天气信息如下：5 月 1 日：晴天；5 月 2 日：小雨；5 月 3 日：大雨。"""
        # 使用 completed_task 工具函数直接返回已完成的任务和 artifact
        event_queue.enqueue_event(
            completed_task(
                context.task_id,
                context.context_id,
                [new_artifact(parts=[Part(root=TextPart(text=text))], name="天气查询结果")],
                [context.message],
            )
        )

    # 取消操作未实现，抛出不支持的操作异常
    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
