from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from agent_executor import FlightAgentExecutor


def main(host: str, port: int):
    # 声明 Agent 能力：支持流式输出
    capabilities = AgentCapabilities(streaming=True)

    # 定义技能1：查询机票信息
    query_flight_skill = AgentSkill(
        id='查询机票信息',
        name='查询机票信息',
        description='给定时间，查询对应的机票信息',
        tags=['查询', '机票'],
        examples=['给我查询5月1日的机票信息'],
    )
    # 定义技能2：预定机票
    book_skill = AgentSkill(
        id='预定机票',
        name='预定机票',
        description='预定机票',
        tags=['预定', '机票'],
        examples=['给我预定5月1日从纽约飞往旧金山的机票'],
    )

    # 创建 Agent Card（Agent 的元数据描述卡片），供客户端发现和调用
    agent_card = AgentCard(
        name='机票 Agent',
        description='提供机票查询和预订功能',
        url=f'http://{host}:{port}',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=capabilities,
        skills=[query_flight_skill, book_skill],
    )

    # 创建请求处理器，绑定执行器和内存任务存储
    request_handler = DefaultRequestHandler(
        agent_executor=FlightAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    # 创建 A2A 服务端应用
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    import uvicorn

    # 启动 HTTP 服务
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == '__main__':
    # 机票 Agent 监听端口 10001
    main("127.0.0.1", 10001)
