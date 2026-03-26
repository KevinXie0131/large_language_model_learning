# 导入 A2A 服务端框架组件
from a2a.server.apps import A2AStarletteApplication  # 基于 Starlette 的 A2A 应用
from a2a.server.request_handlers import DefaultRequestHandler  # 默认请求处理器
from a2a.server.tasks import InMemoryTaskStore  # 内存任务存储（用于管理任务状态）
from a2a.types import (
    AgentCapabilities,  # Agent 能力声明
    AgentCard,          # Agent 名片（描述 Agent 的元信息）
    AgentSkill,         # Agent 技能定义
)

from agent_executor import WeatherAgentExecutor  # 导入自定义的天气 Agent 执行器


def main(host: str, port: int):
    """启动天气 Agent 服务。

    Args:
        host: 服务监听地址。
        port: 服务监听端口。
    """
    # 声明 Agent 的能力（当前不支持流式响应）
    capabilities = AgentCapabilities(streaming=False)

    # 定义技能 1：天气预告
    forecast_skill = AgentSkill(
        id='天气预告',
        name='天气预告',
        description='给出某地的天气预告',
        tags=['天气', '预告'],
        examples=['给我纽约未来 7 天的天气预告'],
    )

    # 定义技能 2：空气质量报告
    air_quality_skill = AgentSkill(
        id='空气质量报告',
        name='空气质量报告',
        description='给出某地当前时间的空气质量报告，不做预告',
        tags=['空气', '质量'],
        examples=['给我纽约当前的空气质量报告'],
    )

    # 创建 Agent 名片（AgentCard），包含 Agent 的基本信息和技能列表
    agent_card = AgentCard(
        name='天气 Agent',
        description='提供天气相关的查询功能',
        url=f'http://{host}:{port}',       # Agent 服务的访问地址
        version='1.0.0',
        defaultInputModes=['text'],         # 默认输入模式：文本
        defaultOutputModes=['text'],        # 默认输出模式：文本
        capabilities=capabilities,
        skills=[forecast_skill, air_quality_skill],  # 注册技能列表
    )

    # 创建请求处理器，绑定 Agent 执行器和任务存储
    request_handler = DefaultRequestHandler(
        agent_executor=WeatherAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # 创建 A2A 应用实例
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    # 使用 uvicorn 启动 HTTP 服务
    import uvicorn
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == '__main__':
    # 入口：在本地 127.0.0.1:10000 启动天气 Agent 服务
    main("127.0.0.1", 10000)
