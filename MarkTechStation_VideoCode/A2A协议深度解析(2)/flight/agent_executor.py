import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (Part, Task, TextPart, UnsupportedOperationError, TaskArtifactUpdateEvent, Artifact)
from a2a.utils import new_task
from a2a.utils.errors import ServerError


# 机票 Agent 的执行器，继承自 AgentExecutor，负责处理机票查询请求
class FlightAgentExecutor(AgentExecutor):

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # 获取当前任务，如果不存在则根据消息创建新任务
        task = context.current_task
        if not task:
            task = new_task(context.message)
            event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.contextId)

        # 生成唯一的 artifact ID，用于标识本次查询结果
        artifact_id = str(uuid.uuid4())

        # 以流式方式分三次发送查询结果（模拟流式输出）
        # 第一块：发送开头文本，append=False 表示这是新 artifact 的第一部分
        event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                taskId=task.id,
                contextId=task.contextId,
                artifact=Artifact(
                    artifactId=artifact_id,
                    parts=[Part(root=TextPart(text="你要查询的机票"))],
                ),
                append=False,
                lastChunk=False
            )
        )
        # 第二块：追加文本，append=True 表示追加到已有 artifact
        event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                taskId=task.id,
                contextId=task.contextId,
                artifact=Artifact(
                    artifactId=artifact_id,
                    parts=[Part(root=TextPart(text="如下："))],
                ),
                append=True,
                lastChunk=False
            )
        )
        # 第三块：发送具体航班信息，lastChunk=True 表示这是最后一部分
        event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                taskId=task.id,
                contextId=task.contextId,
                artifact=Artifact(
                    artifactId=artifact_id,
                    parts=[Part(root=TextPart(text="1. 航班号 FAKE-001，起飞时间 20:00，余票 30 张；2. 航班号 FAKE-002，起飞时间 23:00，余票 50 张"))],
                ),
                append=True,
                lastChunk=True
            )
        )
        # 标记任务完成
        updater.complete()

    # 取消操作未实现，抛出不支持的操作异常
    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
