from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from agent_executor import WeatherAgentExecutor


def main(host: str, port: int):
    # 声明 Agent 能力：不支持流式输出
    capabilities = AgentCapabilities(streaming=False)

    # 定义技能1：天气预告
    forecast_skill = AgentSkill(
        id='天气预告',
        name='天气预告',
        description='给出某地的天气预告',
        tags=['天气', '预告'],
        examples=['给我纽约未来 7 天的天气预告'],
    )
    # 定义技能2：空气质量报告
    air_quality_skill = AgentSkill(
        id='空气质量报告',
        name='空气质量报告',
        description='给出某地当前时间的空气质量报告，不做预告',
        tags=['空气', '质量'],
        examples=['给我纽约当前的空气质量报告'],
    )

    # 创建 Agent Card（Agent 的元数据描述卡片），供客户端发现和调用
    agent_card = AgentCard(
        name='天气 Agent',
        description='提供天气相关的查询功能',
        url=f'http://{host}:{port}',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=capabilities,
        skills=[forecast_skill, air_quality_skill],
    )

    # 创建请求处理器，绑定执行器和内存任务存储
    request_handler = DefaultRequestHandler(
        agent_executor=WeatherAgentExecutor(),
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
    # 天气 Agent 监听端口 10000
    main("127.0.0.1", 10000)
