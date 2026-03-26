# mcp_client.py
# MCP 客户端：通过 MCP（Model Context Protocol）协议与 MCP Server 通信。
# MCP 客户端通过 stdio 传输方式启动并连接 MCP Server 子进程，然后调用服务器上注册的工具。

import asyncio
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """MCP 客户端：支持异步上下文管理器，自动管理与 MCP Server 的连接生命周期。"""

    def __init__(self, command: str, args: List[str]):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.command = command  # 启动 MCP Server 的命令（如 "uv"）
        self.args = args        # 启动命令的参数（如 ["run", "mcp_server.py"]）

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """调用 MCP Server 上注册的工具，返回工具执行结果的文本内容。"""
        call_tool_result = await self.session.call_tool(tool_name, tool_args)
        return call_tool_result.content[0].text

    async def connect_to_server(self):
        """通过 stdio 传输方式连接到 MCP Server，建立双向通信通道。"""
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=None
        )

        # 通过 stdio 启动 MCP Server 子进程，获取读写流
        stdio, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        # 创建 MCP 会话并完成协议握手
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

    async def __aenter__(self):
        """进入异步上下文时自动连接 MCP Server。"""
        await self.connect_to_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出异步上下文时自动清理连接资源。"""
        await self.exit_stack.aclose()


# 使用示例：直接运行本文件可测试 MCP Client 连接 MCP Server 并调用工具
if __name__ == "__main__":

    async def main():

        # 获取与当前脚本同目录下的 mcp_server.py 的绝对地址
        mcp_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_server.py"))

        # 启动 MCP Client 并调用 MCP Tool
        async with MCPClient("uv", ["run", mcp_server_path]) as client:
            result = await client.call_tool("search", { "query": "weather in New York"})
            print(result)

    asyncio.run(main())