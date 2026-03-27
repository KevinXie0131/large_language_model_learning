# Weather Agent

A2A 协议天气 Agent 服务，提供天气预告和空气质量报告功能。

## Build & Run

需要 Python 3.13+，使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

```bash
# 安装依赖
uv sync

# 运行服务（监听 127.0.0.1:10000）
# `.` 表示运行当前目录下的 __main__.py 入口文件
uv run .
```
