# backend.py
# 后端核心逻辑：封装了 LLM 调用、Function Calling 工具执行，以及通过 MCP 协议调用工具的能力。
# 本文件演示了 Function Calling 的完整流程：用户提问 → 模型判断是否需要调用工具 → 执行工具 → 模型生成最终回答。

import asyncio

import requests
import json
from dotenv import load_dotenv
import os

from mcp_client import MCPClient


def get_api_key() -> str:
    """Load the API key from an environment variable."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。")
    return api_key


# 加载 API 密钥和模型配置
OPENROUTER_API_KEY = get_api_key()
MODEL_NAME = "openai/gpt-4o-mini"

# 定义可供模型调用的工具列表（Function Calling 的工具声明）
# 每个工具包含名称、描述和参数的 JSON Schema 定义，模型会根据这些信息决定是否调用工具
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索网络",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的内容"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


class AppLogger:
    def __init__(self):
        """Initialize the logger with a file that will be cleared on startup."""
        self.log_file = "model.log"
        # Clear the log file on startup
        with open(self.log_file, 'w') as f:
            f.write("")

    def log(self, message):
        """Log a message to both file and console."""

        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")


logger = AppLogger()


class LLMProcessor:
    """LLM 处理器：负责与大语言模型交互，处理 Function Calling 流程。"""

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # 维护多轮对话的消息历史
        self.history = []

    def process_user_query(self, query):
        """处理用户查询的主流程：
        1. 将用户消息加入历史
        2. 第一次调用模型，判断是否需要调用工具
        3. 如果需要调用工具，执行工具并将结果返回给模型
        4. 第二次调用模型，生成最终回答
        """

        self.history.append({"role": "user", "content": query})

        first_model_response = self.call_model()

        first_model_message = first_model_response["choices"][0]["message"]
        self.history.append(first_model_message)

        # 检查模型是否需要调用工具
        if "tool_calls" in first_model_message and first_model_message["tool_calls"]:
            tool_call = first_model_message["tool_calls"][0]
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            result = self.execute_tool(tool_name, tool_args)

            self.history.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": result
            })

            second_response_data = self.call_model_after_tool_execution()

            final_message = second_response_data["choices"][0]["message"]
            self.history.append(final_message)

            return {
                "tool_name": tool_name,
                "tool_parameters": tool_args,
                "tool_executed": True,
                "tool_result": result,
                "final_response": final_message["content"],
            }
        else:
            return {
                "final_response": first_model_message["content"],
            }

    def execute_tool(self, function_name, args):
        """直接执行工具（不通过 MCP 协议），根据工具名称分发到对应的处理函数。"""
        if function_name == "search":
            # 正常情况下，这里应该调用相关 API 做搜索，为了减少代码的复杂度，
            # 这里我们返回一段假的工具执行结果，用以测试
            return "纽约市今天的天气是晴天，明天的天气是多云。"
        else:
            raise ValueError(f"未知的工具名称：{function_name}")

    def call_model(self):
        """第一次调用模型：发送用户消息和工具定义，让模型决定是否需要调用工具。"""
        request_body = {
            "model": MODEL_NAME,
            "messages": self.history,
            "tools": TOOLS,
            "stream": False,
        }

        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=request_body
        )

        logger.log(f"第一次模型请求：\n{json.dumps(request_body, indent=2, ensure_ascii=False)}\n")
        logger.log(f"第一次模型返回：\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}\n")

        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        return response.json()

    def call_model_after_tool_execution(self):
        """第二次调用模型：工具执行完毕后，将工具结果发送给模型，让模型生成最终回答。"""
        second_request_body = {
            "model": MODEL_NAME,
            "messages": self.history,
            "tools": TOOLS,
        }

        # Make the second POST request
        second_response = requests.post(
            self.base_url,
            headers=self.headers,
            json=second_request_body
        )

        logger.log(f"第二次模型请求：\n{json.dumps(second_request_body, indent=2, ensure_ascii=False)}\n")
        logger.log(f"第二次模型返回：\n{json.dumps(second_response.json(), indent=2, ensure_ascii=False)}\n")

        # Check if the request was successful
        if second_response.status_code != 200:
            raise Exception(f"API request failed with status {second_response.status_code}: {second_response.text}")

        # Parse the second response
        return second_response.json()

    def execute_tool_with_mcp(self, function_name, args):
        """通过 MCP 协议执行工具（同步包装器）。
        与 execute_tool 不同，这里通过 MCP Client 连接 MCP Server 来执行工具，
        体现了 MCP 协议将工具执行解耦到独立服务的设计思想。
        """
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(self.execute_tool_with_mcp_async(function_name, args))

    async def execute_tool_with_mcp_async(self, function_name, args):
        # 获取与当前脚本同目录下的 mcp_server.py 的绝对地址
        mcp_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_server.py"))

        # 启动 MCP Client 并调用 MCP Tool
        async with MCPClient("uv", ["run", mcp_server_path]) as client:
            return await client.call_tool(function_name, args)

