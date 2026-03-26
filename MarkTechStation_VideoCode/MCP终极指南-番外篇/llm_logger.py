# LLM API Logger - A FastAPI proxy that logs all LLM API requests and responses
# LLM API 日志记录器 - 一个用于记录所有 LLM API 请求和响应的 FastAPI 代理
# This service sits between the client and OpenRouter API, capturing traffic for debugging
# 该服务位于客户端和 OpenRouter API 之间，捕获流量用于调试

import httpx
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse


class AppLogger:
    def __init__(self, log_file="llm.log"):
        """Initialize the logger with a file that will be cleared on startup.
        初始化日志记录器，启动时清空日志文件。"""
        self.log_file = log_file
        # Clear the log file on startup / 启动时清空日志文件
        with open(self.log_file, 'w') as f:
            f.write("")

    def log(self, message):
        """Log a message to both file and console.
        将消息同时记录到文件和控制台。"""

        # Log to file / 记录到文件
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

        # Log to console / 输出到控制台
        print(message)


# Create FastAPI app and initialize logger (log file is cleared on each startup)
# 创建 FastAPI 应用并初始化日志记录器（每次启动时清空日志文件）
app = FastAPI(title="LLM API Logger")
logger = AppLogger("llm.log")


# Proxy endpoint: receives chat completion requests, logs them,
# forwards to OpenRouter API, and streams the response back while logging each line
# 代理端点：接收聊天补全请求并记录，转发到 OpenRouter API，并在记录每一行的同时流式返回响应
@app.post("/chat/completions")
async def proxy_request(request: Request):

    # Read and log the incoming request body / 读取并记录传入的请求体
    body_bytes = await request.body()
    body_str = body_bytes.decode('utf-8')
    logger.log(f"模型请求：{body_str}")
    body = await request.json()

    logger.log("模型返回：\n")

    # Stream the response from OpenRouter API back to the client
    # 将 OpenRouter API 的响应流式传输回客户端
    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            # Forward the request to OpenRouter with the same auth token
            # 使用相同的认证令牌将请求转发到 OpenRouter
            async with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "Authorization": request.headers.get("Authorization"),
                    },
            ) as response:
                # Log and yield each line of the streamed response
                # 记录并逐行返回流式响应
                async for line in response.aiter_lines():
                    logger.log(line)
                    yield f"{line}\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Run the server on port 8000 when executed directly
# 直接运行时在端口 8000 启动服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
