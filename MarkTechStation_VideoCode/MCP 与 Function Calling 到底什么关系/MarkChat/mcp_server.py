# mcp_server.py
# MCP 服务端：使用 FastMCP 框架注册工具，通过 stdio 传输方式对外提供服务。
# MCP Server 将工具能力独立封装为服务，任何 MCP Client 都可以连接并调用这些工具，
# 实现了工具能力与 LLM 应用的解耦。

from mcp.server.fastmcp import FastMCP


# 初始化 FastMCP 服务器实例，指定服务名称
mcp = FastMCP("search_mcp_server", log_level="ERROR")


# 使用 @mcp.tool() 装饰器将函数注册为 MCP 工具，客户端可通过工具名称远程调用
@mcp.tool()
async def search(query: str) -> str:
    """搜索网络

    Args:
        query: 搜索内容
    """
    # 正常情况下，这里应该调用相关 API 做搜索，为了减少代码的复杂度，
    # 这里我们返回一段假的工具执行结果，用以测试
    return "来自 MCP Server 的答案：纽约市今天的天气是晴天，明天的天气是多云。"


if __name__ == "__main__":
    # 启动 MCP Server，使用 stdio 作为传输方式（通过标准输入/输出与客户端通信）
    mcp.run(transport='stdio')