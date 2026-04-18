# CLAUDE.md

## Project Overview

**nanobot** (`nanobot-ai` on PyPI, v0.1.4.post6) is an ultra-lightweight personal AI assistant framework inspired by [OpenClaw](https://github.com/openclaw/openclaw). It delivers core agent functionality with 99% fewer lines of code, providing multi-channel chat integration across 12+ platforms with 25+ LLM provider support. Licensed under MIT.

---

## Tech Stack, Frameworks & Versions

### Languages

| Language | Version | Usage |
|----------|---------|-------|
| Python | >= 3.11 (CI: 3.11, 3.12, 3.13) | Primary language for agent core, CLI, channels, providers |
| TypeScript | ^5.4.0 | WhatsApp bridge |
| Node.js | >= 20.0.0 | WhatsApp bridge runtime |

### Core Frameworks & Libraries (Python)

| Library | Version Constraint | Purpose |
|---------|--------------------|---------|
| typer | >=0.20.0, <1.0.0 | CLI framework |
| rich | >=14.0.0, <15.0.0 | Terminal formatting & output |
| prompt-toolkit | >=3.0.50, <4.0.0 | Interactive REPL (history, highlighting, paste mode) |
| questionary | >=2.0.0, <3.0.0 | Interactive prompts for onboarding wizard |
| pydantic | >=2.12.0, <3.0.0 | Configuration schema validation |
| pydantic-settings | >=2.12.0, <3.0.0 | Env var / JSON config merging |
| anthropic | >=0.45.0, <1.0.0 | Anthropic (Claude) native SDK |
| openai | >=2.8.0 | OpenAI-compatible provider SDK |
| mcp | >=1.26.0, <2.0.0 | Model Context Protocol client |
| httpx | >=0.28.0, <1.0.0 | Async HTTP client |
| websockets | >=16.0, <17.0 | WebSocket protocol support |
| websocket-client | >=1.9.0, <2.0.0 | Synchronous WebSocket client |
| python-socketio | >=5.16.0, <6.0.0 | Socket.IO client (Discord) |
| tiktoken | >=0.12.0, <1.0.0 | Token counting |
| loguru | >=0.7.3, <1.0.0 | Structured logging |
| croniter | >=6.0.0, <7.0.0 | Cron expression parsing |
| ddgs | >=9.5.5, <10.0.0 | DuckDuckGo search |
| readability-lxml | >=0.8.4, <1.0.0 | Web content extraction |
| json-repair | >=0.57.0, <1.0.0 | Malformed JSON recovery |
| chardet | >=3.0.2, <6.0.0 | Character encoding detection |
| msgpack | >=1.1.0, <2.0.0 | Binary serialization |
| oauth-cli-kit | >=0.1.3, <1.0.0 | OAuth flow for CLI |

### Channel SDKs (Python)

| Library | Version Constraint | Channel |
|---------|--------------------|---------|
| python-telegram-bot[socks] | >=22.6, <23.0 | Telegram (with SOCKS proxy) |
| slack-sdk | >=3.39.0, <4.0.0 | Slack |
| slackify-markdown | >=0.2.0, <1.0.0 | Slack markdown conversion |
| dingtalk-stream | >=0.24.0, <1.0.0 | DingTalk |
| lark-oapi | >=1.5.0, <2.0.0 | Feishu (Lark) |
| qq-botpy | >=1.2.0, <2.0.0 | QQ |
| socksio | >=1.0.0, <2.0.0 | SOCKS proxy support |
| python-socks[asyncio] | >=2.8.0, <3.0.0 | Async SOCKS proxy |

### Optional Extras (Python)

| Extra | Libraries | Purpose |
|-------|-----------|---------|
| `wecom` | wecom-aibot-sdk-python >=0.1.5 | WeCom (WeChat Work) |
| `weixin` | qrcode[pil] >=8.0, pycryptodome >=3.20.0 | WeChat |
| `matrix` | matrix-nio[e2e] >=0.25.2, mistune >=3.0.0, nh3 >=0.2.17 | Matrix |
| `langsmith` | langsmith >=0.1.0 | LangSmith tracing |

### Node.js Dependencies (WhatsApp Bridge)

| Library | Version | Purpose |
|---------|---------|---------|
| @whiskeysockets/baileys | 7.0.0-rc.9 | WhatsApp Web API |
| ws | ^8.17.1 | WebSocket server (bridge <-> Python) |
| qrcode-terminal | ^0.12.0 | QR code login display |
| pino | ^9.0.0 | Logging |
| typescript | ^5.4.0 | Build tooling (devDependency) |

### Build & Dev Tooling

| Tool | Version | Purpose |
|------|---------|---------|
| uv | latest | Python package manager & installer |
| hatchling | latest | Python build backend |
| ruff | >=0.1.0 | Linter & formatter (rules: E, F, I, N, W; line-length: 100; target: py311) |
| pytest | >=9.0.0, <10.0.0 | Test runner |
| pytest-asyncio | >=1.3.0, <2.0.0 | Async test support |
| pytest-cov | >=6.0.0, <7.0.0 | Coverage reporting |
| Docker | - | Containerization (python3.12-bookworm-slim base) |
| docker-compose | - | Multi-service orchestration |
| GitHub Actions | - | CI (Python 3.11, 3.12, 3.13) |

---

## Features & Functionality

### Agent System
- **Agent Loop** (`agent/loop.py`): Async message processing with configurable max iterations (default 40), streaming, token counting, rate limiting
- **Subagents** (`agent/subagent.py`): Spawn autonomous background agents for long-running tasks; results announced back via message bus
- **Runner & Hooks** (`agent/runner.py`, `agent/hook.py`): Core LLM loop with tool execution, lifecycle callbacks, configurable concurrency
- **Context Builder** (`agent/context.py`): Assembles system prompt from identity, bootstrap files (AGENTS.md, SOUL.md, USER.md, TOOLS.md), memory, active skills, and platform policy

### Memory System
- **Two-layer memory** (`agent/memory.py`):
  - `MEMORY.md` — Long-term facts consolidated by the LLM
  - `HISTORY.md` — Grep-searchable conversation log with timestamps
- **MemoryConsolidator**: Auto-archives old messages when context window exceeds safe threshold
- Graceful fallback (raw-dump) after 3 consecutive consolidation failures

### Built-in Tools
- **File ops:** ReadFile, WriteFile, EditFile, ListDir (workspace-restricted)
- **Shell exec:** ExecTool with timeout and dangerous-command blocking
- **Web search:** Multi-provider (Brave, Tavily, DuckDuckGo, SearXNG, Jina)
- **Web fetch:** URL content retrieval with proxy support
- **Message:** Send messages within a turn
- **Spawn:** Trigger subagent creation
- **Cron:** Schedule recurring/one-time tasks
- **MCP tools:** Dynamic tools from configured MCP servers (stdio, SSE, streamable HTTP)

### Skills System
- Markdown-based (SKILL.md) with metadata; loaded from workspace and built-in directories
- **Built-in skills:** clawhub (skill registry), cron, github, memory, skill-creator, summarize, tmux, weather
- Workspace skills override built-in skills; can be marked always-available or on-demand

### MCP (Model Context Protocol)
- Supports stdio, SSE, and streamable HTTP transports
- Per-server config: command, args, env, URL, headers, tool timeout, enabled tools filter
- Lazy initialization of MCP connections

### Supported Chat Channels (12+)

| Channel | SDK/Integration |
|---------|----------------|
| Telegram | python-telegram-bot (with SOCKS proxy) |
| Slack | slack-sdk + slackify-markdown |
| Discord | discord.py via socketio |
| DingTalk | dingtalk-stream |
| Feishu (Lark) | lark-oapi |
| QQ | qq-botpy |
| WeChat | Custom (qrcode, pycryptodome) — optional extra |
| WeCom | wecom-aibot-sdk-python — optional extra |
| WhatsApp | Node.js bridge (Baileys + WebSocket) |
| Matrix | matrix-nio[e2e] — optional extra |
| Email | Built-in |
| MoChat | Built-in |

### Supported LLM Providers (25+)

**Provider backends:** Anthropic (native), OpenAI-compatible, Azure OpenAI, OpenAI Codex (OAuth)

**Services:** Anthropic, OpenAI, OpenRouter, DeepSeek, Groq, ZhiPu, DashScope (Qwen), vLLM, Ollama, OVMS (OpenVINO), Gemini, Moonshot/Kimi, MiniMax, Mistral, StepFun, AiHubMix, SiliconFlow, VolcEngine, VolcEngine Coding Plan, BytePlus, BytePlus Coding Plan, OpenAI Codex, GitHub Copilot, plus any custom OpenAI-compatible endpoint.

Auto-detection selects the provider based on model name prefix/keywords, with fallback to first configured key.

### Streaming
- End-to-end streaming output across all channels
- Progress streaming and tool-call hints (configurable)
- Prompt caching for Anthropic provider
- Extended thinking mode support (reasoning_effort: low/medium/high)

### CLI
- Interactive REPL with prompt-toolkit (history, syntax highlighting, paste mode)
- Commands: `/status`, `/memory`, `/cron`, `/stop`, etc.
- Onboarding wizard (`nanobot onboard --wizard`) with provider selection and model autocomplete
- Entry point: `nanobot = nanobot.cli.commands:app`

---

## System Design

### Architecture Overview

```
                           +-----------------------+
                           |    nanobot gateway     |
                           |    (asyncio.run)       |
                           +----------+------------+
                                      |
            +-------------------------+-------------------------+
            |              |                |                    |
    +-------v-------+  +--v----------+  +--v----------+  +-----v--------+
    |  AgentLoop    |  | ChannelMgr  |  | CronService |  | HeartbeatSvc |
    | (per-session  |  | (12+ chat   |  | (at/every/  |  | (30-min      |
    |  serial,      |  |  platforms) |  |  cron expr) |  |  wake-ups)   |
    |  cross-session|  +------+------+  +------+------+  +------+-------+
    |  concurrent)  |         |                |                |
    +-------+-------+    push | consume   inject|           inject|
            |            msgs |  msgs      msgs |            msgs |
            v                 v                 v                 v
    +-------+--------------------------------------------------+---+
    |                       MessageBus                              |
    |  inbound: asyncio.Queue[InboundMessage]   (channels -> agent) |
    |  outbound: asyncio.Queue[OutboundMessage] (agent -> channels) |
    +--------------------------------------------------------------+
```

### Key Design Decisions

1. **Event-driven, fully async (asyncio):** All I/O is non-blocking. The entire system runs inside a single `asyncio.run()` event loop.

2. **Message bus decoupling:** Channels and agent core communicate exclusively through the `MessageBus` (two `asyncio.Queue` instances — inbound and outbound). Channels push user messages to `inbound`; the agent processes them and pushes responses to `outbound`; the `ChannelManager` dispatcher consumes `outbound` and delivers to the correct channel.

3. **Per-session serial, cross-session concurrent:** Each session (channel + chat ID) holds an `asyncio.Lock` ensuring messages are processed one at a time. Different sessions run concurrently, gated by a global `asyncio.Semaphore` (default 3, configurable via `NANOBOT_MAX_CONCURRENT_REQUESTS`).

4. **No embedded HTTP server:** Channels use platform-native APIs (WebSocket, Socket Mode, polling, long-polling). The gateway port (18790) is exposed for channel-specific webhooks where needed, not as a general HTTP server.

5. **File-based storage (no database):**
   - **Sessions:** Append-only JSONL files at `workspace/sessions/{channel}_{chat_id}.jsonl`. First line is metadata; subsequent lines are messages. Append-only design optimizes LLM prompt caching.
   - **Config:** JSON (`~/.nanobot/config.json`) merged with `NANOBOT_*` environment variables via `pydantic-settings`.
   - **Memory:** Markdown files (`MEMORY.md`, `HISTORY.md`) in workspace.
   - **Cron jobs:** `workspace/cron/jobs.json` with per-job run history.
   - **Media:** Channel-specific subdirectories under `~/.nanobot/media/`.

6. **Two-layer memory:** `MEMORY.md` stores LLM-consolidated long-term facts. `HISTORY.md` is a grep-searchable timestamped conversation log. The `MemoryConsolidator` auto-archives when context exceeds the safe threshold, with graceful fallback after 3 consecutive failures.

7. **Context assembly pipeline:** The `ContextBuilder` composes the system prompt from: identity config -> bootstrap templates (AGENTS.md, SOUL.md, USER.md, TOOLS.md) -> memory -> active skills -> platform policy -> tool definitions (including MCP tools).

8. **Agent loop with tool iteration:** The agent loop runs up to 40 iterations (configurable) per message, executing tool calls and feeding results back. Tool results are truncated at 16,000 characters.

### Project Structure

```
nanobot/
  agent/       - Core agent loop, context builder, memory, runner, subagent, hooks, tools/
  channels/    - Chat platform integrations (12+ platforms)
  providers/   - LLM provider integrations (4 provider backends, 25+ services)
  cli/         - CLI commands, onboarding wizard, streaming output
  config/      - Pydantic schema, config loader, path resolution
  session/     - Conversation session manager (JSONL storage)
  skills/      - Built-in skills (clawhub, cron, github, memory, skill-creator, summarize, tmux, weather)
  command/     - Command routing and built-in commands
  cron/        - Task scheduling service (at/every/cron expressions)
  bus/         - Async event bus and message queue
  heartbeat/   - Periodic task wake-ups (30-min default)
  security/    - Network security utilities
  utils/       - Helpers and expression evaluation
  templates/   - Prompt templates (AGENTS.md, SOUL.md, USER.md, TOOLS.md, HEARTBEAT.md)
bridge/        - Node.js WhatsApp bridge (Baileys + WebSocket)
tests/         - pytest tests organized by module
docs/          - Documentation site
```

### Configuration

Root config (`~/.nanobot/config.json`) with Pydantic schema:
- `agents.defaults` — model, provider, workspace, context window (default 65536), max tokens, temperature, max iterations (default 40), reasoning effort, timezone
- `channels` — per-channel settings, streaming, tool hints, delivery retries
- `providers` — API keys and base URLs for 25+ providers
- `gateway` — host (0.0.0.0), port (18790), heartbeat settings
- `tools` — web search/proxy, exec tool, workspace restriction, MCP servers

Environment variable override: `NANOBOT_<SECTION>__<KEY>` (double underscore delimiter).

---

## How to Build & Run Locally

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Node.js >= 20 (only needed for WhatsApp bridge)
- Git

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/nanobot-ai/nanobot.git
cd nanobot

# Install Python dependencies
uv install

# (Optional) Install optional extras
uv install --extra wecom
uv install --extra weixin
uv install --extra matrix

# (Optional) Install dev dependencies
uv install --extra dev
```

### 2. Initial Configuration

```bash
# Quick setup — creates ~/.nanobot/config.json
uv run nanobot onboard

# Interactive wizard — select providers, models, channels
uv run nanobot onboard --wizard
```

Or set environment variables directly:
```bash
export NANOBOT_PROVIDERS__ANTHROPIC__API_KEY="sk-ant-..."
export NANOBOT_PROVIDERS__OPENAI__API_KEY="sk-..."
export NANOBOT_AGENTS__DEFAULTS__MODEL="anthropic/claude-sonnet-4-20250514"
```

### 3. Run

```bash
# Interactive CLI (REPL mode)
uv run nanobot

# Start gateway (all channels + cron + heartbeat)
uv run nanobot gateway

# Start gateway on custom port
uv run nanobot gateway --port 8080

# Check status
uv run nanobot status

# Authenticate a channel
uv run nanobot channels login telegram
```

### 4. Build & Run WhatsApp Bridge (Optional)

```bash
cd bridge
npm install
npm run build
npm start      # or: npm run dev
```

### 5. Docker Deployment

```bash
# Build and run gateway
docker-compose up -d nanobot-gateway

# Run CLI in container
docker-compose run --rm nanobot-cli

# Or build manually
docker build -t nanobot .
docker run -v ~/.nanobot:/root/.nanobot -p 18790:18790 nanobot gateway
```

### 6. Development Workflow

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=nanobot

# Lint
ruff check .

# Format
ruff format .
```

---
---

# CLAUDE.md (中文版)

## 项目概述

**nanobot**（PyPI 包名 `nanobot-ai`，版本 v0.1.4.post6）是一个超轻量级的个人 AI 助手框架，灵感来源于 [OpenClaw](https://github.com/openclaw/openclaw)。以 99% 更少的代码量实现核心 Agent 功能，支持 12+ 聊天平台的多渠道集成和 25+ 大语言模型服务商。MIT 许可证。

---

## 技术栈、框架与版本

### 编程语言

| 语言 | 版本 | 用途 |
|------|------|------|
| Python | >= 3.11（CI 测试：3.11、3.12、3.13） | 主语言，用于 Agent 核心、CLI、渠道、服务商集成 |
| TypeScript | ^5.4.0 | WhatsApp 桥接服务 |
| Node.js | >= 20.0.0 | WhatsApp 桥接运行时 |

### 核心框架与库（Python）

| 库 | 版本约束 | 用途 |
|----|----------|------|
| typer | >=0.20.0, <1.0.0 | CLI 框架 |
| rich | >=14.0.0, <15.0.0 | 终端格式化与输出美化 |
| prompt-toolkit | >=3.0.50, <4.0.0 | 交互式 REPL（历史记录、语法高亮、粘贴模式） |
| questionary | >=2.0.0, <3.0.0 | 交互式引导向导 |
| pydantic | >=2.12.0, <3.0.0 | 配置模型校验 |
| pydantic-settings | >=2.12.0, <3.0.0 | 环境变量 / JSON 配置合并 |
| anthropic | >=0.45.0, <1.0.0 | Anthropic (Claude) 原生 SDK |
| openai | >=2.8.0 | OpenAI 兼容服务商 SDK |
| mcp | >=1.26.0, <2.0.0 | Model Context Protocol 客户端 |
| httpx | >=0.28.0, <1.0.0 | 异步 HTTP 客户端 |
| websockets | >=16.0, <17.0 | WebSocket 协议支持 |
| websocket-client | >=1.9.0, <2.0.0 | 同步 WebSocket 客户端 |
| python-socketio | >=5.16.0, <6.0.0 | Socket.IO 客户端（Discord） |
| tiktoken | >=0.12.0, <1.0.0 | Token 计数 |
| loguru | >=0.7.3, <1.0.0 | 结构化日志 |
| croniter | >=6.0.0, <7.0.0 | Cron 表达式解析 |
| ddgs | >=9.5.5, <10.0.0 | DuckDuckGo 搜索 |
| readability-lxml | >=0.8.4, <1.0.0 | 网页正文提取 |
| json-repair | >=0.57.0, <1.0.0 | 异常 JSON 修复 |
| chardet | >=3.0.2, <6.0.0 | 字符编码检测 |
| msgpack | >=1.1.0, <2.0.0 | 二进制序列化 |
| oauth-cli-kit | >=0.1.3, <1.0.0 | CLI OAuth 认证流程 |

### 渠道 SDK（Python）

| 库 | 版本约束 | 渠道 |
|----|----------|------|
| python-telegram-bot[socks] | >=22.6, <23.0 | Telegram（支持 SOCKS 代理） |
| slack-sdk | >=3.39.0, <4.0.0 | Slack |
| slackify-markdown | >=0.2.0, <1.0.0 | Slack Markdown 转换 |
| dingtalk-stream | >=0.24.0, <1.0.0 | 钉钉 |
| lark-oapi | >=1.5.0, <2.0.0 | 飞书 |
| qq-botpy | >=1.2.0, <2.0.0 | QQ |
| socksio | >=1.0.0, <2.0.0 | SOCKS 代理支持 |
| python-socks[asyncio] | >=2.8.0, <3.0.0 | 异步 SOCKS 代理 |

### 可选扩展（Python）

| 扩展名 | 依赖库 | 用途 |
|--------|--------|------|
| `wecom` | wecom-aibot-sdk-python >=0.1.5 | 企业微信 |
| `weixin` | qrcode[pil] >=8.0, pycryptodome >=3.20.0 | 微信 |
| `matrix` | matrix-nio[e2e] >=0.25.2, mistune >=3.0.0, nh3 >=0.2.17 | Matrix |
| `langsmith` | langsmith >=0.1.0 | LangSmith 追踪 |

### Node.js 依赖（WhatsApp 桥接）

| 库 | 版本 | 用途 |
|----|------|------|
| @whiskeysockets/baileys | 7.0.0-rc.9 | WhatsApp Web API |
| ws | ^8.17.1 | WebSocket 服务（桥接 <-> Python） |
| qrcode-terminal | ^0.12.0 | 终端二维码登录 |
| pino | ^9.0.0 | 日志 |
| typescript | ^5.4.0 | 构建工具（开发依赖） |

### 构建与开发工具

| 工具 | 版本 | 用途 |
|------|------|------|
| uv | latest | Python 包管理器 |
| hatchling | latest | Python 构建后端 |
| ruff | >=0.1.0 | 代码检查与格式化（规则：E, F, I, N, W；行宽：100；目标：py311） |
| pytest | >=9.0.0, <10.0.0 | 测试运行器 |
| pytest-asyncio | >=1.3.0, <2.0.0 | 异步测试支持 |
| pytest-cov | >=6.0.0, <7.0.0 | 覆盖率报告 |
| Docker | - | 容器化（基础镜像 python3.12-bookworm-slim） |
| docker-compose | - | 多服务编排 |
| GitHub Actions | - | CI（Python 3.11、3.12、3.13） |

---

## 功能与特性

### Agent 系统
- **Agent 循环** (`agent/loop.py`)：异步消息处理，可配置最大迭代次数（默认 40 次），支持流式输出、Token 计数、速率限制
- **子 Agent** (`agent/subagent.py`)：可生成自主后台 Agent 执行长时间任务，结果通过消息总线回传
- **Runner 与钩子** (`agent/runner.py`、`agent/hook.py`)：核心 LLM 循环，支持工具执行、生命周期回调、可配置并发
- **上下文构建器** (`agent/context.py`)：从身份配置、引导文件（AGENTS.md、SOUL.md、USER.md、TOOLS.md）、记忆、活跃技能和平台策略组装系统提示词

### 记忆系统
- **双层记忆** (`agent/memory.py`)：
  - `MEMORY.md` — 由 LLM 整合的长期事实记忆
  - `HISTORY.md` — 带时间戳的可搜索对话日志
- **记忆整合器 (MemoryConsolidator)**：当上下文窗口超出安全阈值时自动归档旧消息
- 连续 3 次整合失败后优雅降级（原始转储）

### 内置工具
- **文件操作：** ReadFile、WriteFile、EditFile、ListDir（受工作区限制）
- **Shell 执行：** ExecTool，支持超时和危险命令拦截
- **网页搜索：** 多服务商（Brave、Tavily、DuckDuckGo、SearXNG、Jina）
- **网页抓取：** URL 内容获取，支持代理
- **消息：** 在一轮对话中发送消息
- **Spawn：** 触发子 Agent 创建
- **定时任务：** 调度周期性/一次性任务
- **MCP 工具：** 来自已配置 MCP 服务器的动态工具（stdio、SSE、streamable HTTP）

### 技能系统
- 基于 Markdown（SKILL.md）的技能定义，带元数据；从工作区和内置目录加载
- **内置技能：** clawhub（技能注册中心）、cron、github、memory、skill-creator、summarize、tmux、weather
- 工作区技能可覆盖内置技能；可标记为始终可用或按需加载

### MCP（Model Context Protocol）
- 支持 stdio、SSE 和 streamable HTTP 传输方式
- 每个服务器可独立配置：命令、参数、环境变量、URL、请求头、工具超时、启用工具过滤
- 延迟初始化 MCP 连接

### 支持的聊天渠道（12+）

| 渠道 | SDK/集成方式 |
|------|-------------|
| Telegram | python-telegram-bot（支持 SOCKS 代理） |
| Slack | slack-sdk + slackify-markdown |
| Discord | discord.py via socketio |
| 钉钉 | dingtalk-stream |
| 飞书 | lark-oapi |
| QQ | qq-botpy |
| 微信 | 自定义实现（qrcode、pycryptodome）— 可选扩展 |
| 企业微信 | wecom-aibot-sdk-python — 可选扩展 |
| WhatsApp | Node.js 桥接（Baileys + WebSocket） |
| Matrix | matrix-nio[e2e] — 可选扩展 |
| 邮件 | 内置 |
| MoChat | 内置 |

### 支持的 LLM 服务商（25+）

**服务商后端：** Anthropic（原生）、OpenAI 兼容、Azure OpenAI、OpenAI Codex（OAuth）

**服务商列表：** Anthropic、OpenAI、OpenRouter、DeepSeek、Groq、智谱、通义千问 (DashScope)、vLLM、Ollama、OVMS (OpenVINO)、Gemini、Moonshot/Kimi、MiniMax、Mistral、阶跃星辰 (StepFun)、AiHubMix、硅基流动 (SiliconFlow)、火山引擎、火山引擎 Coding Plan、BytePlus、BytePlus Coding Plan、OpenAI Codex、GitHub Copilot，以及任意 OpenAI 兼容端点。

自动检测根据模型名称前缀/关键词选择服务商，回退至第一个已配置的 API 密钥。

### 流式输出
- 端到端跨所有渠道的流式输出
- 进度流和工具调用提示（可配置）
- Anthropic 服务商支持 Prompt 缓存
- 扩展思考模式支持（reasoning_effort: low/medium/high）

### 命令行界面
- 基于 prompt-toolkit 的交互式 REPL（历史记录、语法高亮、粘贴模式）
- 命令：`/status`、`/memory`、`/cron`、`/stop` 等
- 引导向导（`nanobot onboard --wizard`），支持服务商选择和模型自动补全
- 入口点：`nanobot = nanobot.cli.commands:app`

---

## 系统设计

### 架构总览

```
                           +-----------------------+
                           |    nanobot gateway     |
                           |    (asyncio.run)       |
                           +----------+------------+
                                      |
            +-------------------------+-------------------------+
            |              |                |                    |
    +-------v-------+  +--v----------+  +--v----------+  +-----v--------+
    |  AgentLoop    |  | ChannelMgr  |  | CronService |  | HeartbeatSvc |
    | (会话内串行,  |  | (12+ 聊天   |  | (at/every/  |  | (30 分钟     |
    |  跨会话并发)  |  |  平台)      |  |  cron 表达式)|  |  定期唤醒)   |
    +-------+-------+  +------+------+  +------+------+  +------+-------+
            |                 |                |                |
            v                 v                v                v
    +--------------------------------------------------------------+
    |                       MessageBus 消息总线                      |
    |  inbound: asyncio.Queue[InboundMessage]   (渠道 -> Agent)     |
    |  outbound: asyncio.Queue[OutboundMessage] (Agent -> 渠道)     |
    +--------------------------------------------------------------+
```

### 核心设计决策

1. **事件驱动，全异步（asyncio）：** 所有 I/O 均为非阻塞。整个系统运行在单一 `asyncio.run()` 事件循环中。

2. **消息总线解耦：** 渠道与 Agent 核心仅通过 `MessageBus` 通信（两个 `asyncio.Queue` 实例 — inbound 和 outbound）。渠道将用户消息推入 `inbound`，Agent 处理后将响应推入 `outbound`，`ChannelManager` 调度器消费 `outbound` 并投递到正确渠道。

3. **会话内串行，跨会话并发：** 每个会话（渠道 + 聊天 ID）持有一个 `asyncio.Lock`，确保消息逐条处理。不同会话并发执行，受全局 `asyncio.Semaphore` 限制（默认 3，可通过 `NANOBOT_MAX_CONCURRENT_REQUESTS` 配置）。

4. **无内嵌 HTTP 服务器：** 渠道使用各平台原生 API（WebSocket、Socket Mode、轮询、长轮询）。网关端口（18790）仅在需要时为特定渠道的 Webhook 回调暴露。

5. **基于文件的存储（无数据库）：**
   - **会话：** 追加写入的 JSONL 文件，路径为 `workspace/sessions/{channel}_{chat_id}.jsonl`。第一行为元数据，后续行为消息。追加写入设计优化了 LLM Prompt 缓存。
   - **配置：** JSON 文件（`~/.nanobot/config.json`），通过 `pydantic-settings` 与 `NANOBOT_*` 环境变量合并。
   - **记忆：** 工作区中的 Markdown 文件（`MEMORY.md`、`HISTORY.md`）。
   - **定时任务：** `workspace/cron/jobs.json`，含每个任务的执行历史。
   - **媒体文件：** `~/.nanobot/media/` 下按渠道分子目录。

6. **双层记忆系统：** `MEMORY.md` 存储 LLM 整合的长期事实。`HISTORY.md` 是可 grep 搜索的带时间戳对话日志。`MemoryConsolidator` 在上下文超出安全阈值时自动归档，连续 3 次失败后优雅降级。

7. **上下文组装管线：** `ContextBuilder` 按以下顺序组装系统提示词：身份配置 -> 引导模板（AGENTS.md、SOUL.md、USER.md、TOOLS.md）-> 记忆 -> 活跃技能 -> 平台策略 -> 工具定义（含 MCP 工具）。

8. **Agent 循环与工具迭代：** Agent 循环每条消息最多运行 40 次迭代（可配置），执行工具调用并将结果反馈回 LLM。工具结果截断上限为 16,000 字符。

### 项目结构

```
nanobot/
  agent/       - 核心 Agent 循环、上下文构建、记忆、运行器、子 Agent、钩子、工具
  channels/    - 聊天平台集成（12+ 平台）
  providers/   - LLM 服务商集成（4 种后端、25+ 服务）
  cli/         - CLI 命令、引导向导、流式输出
  config/      - Pydantic 模型、配置加载器、路径解析
  session/     - 会话管理器（JSONL 存储）
  skills/      - 内置技能（clawhub、cron、github、memory、skill-creator、summarize、tmux、weather）
  command/     - 命令路由与内置命令
  cron/        - 定时任务调度服务（at/every/cron 表达式）
  bus/         - 异步事件总线与消息队列
  heartbeat/   - 定期任务唤醒（默认 30 分钟）
  security/    - 网络安全工具
  utils/       - 辅助工具与表达式求值
  templates/   - 提示词模板（AGENTS.md、SOUL.md、USER.md、TOOLS.md、HEARTBEAT.md）
bridge/        - Node.js WhatsApp 桥接（Baileys + WebSocket）
tests/         - 按模块组织的 pytest 测试
docs/          - 文档站点
```

### 配置

根配置文件（`~/.nanobot/config.json`），基于 Pydantic 模型：
- `agents.defaults` — 模型、服务商、工作区、上下文窗口（默认 65536）、最大 Token 数、温度、最大迭代次数（默认 40）、推理强度、时区
- `channels` — 每个渠道的设置、流式输出、工具提示、投递重试
- `providers` — 25+ 服务商的 API 密钥和基础 URL
- `gateway` — 主机（0.0.0.0）、端口（18790）、心跳设置
- `tools` — 网页搜索/代理、执行工具、工作区限制、MCP 服务器

环境变量覆盖：`NANOBOT_<SECTION>__<KEY>`（双下划线分隔）。

---

## 本地构建与运行

### 前置条件

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- Node.js >= 20（仅 WhatsApp 桥接需要）
- Git

### 1. 克隆与安装依赖

```bash
git clone https://github.com/nanobot-ai/nanobot.git
cd nanobot

# 安装 Python 依赖
uv install

# （可选）安装可选扩展
uv install --extra wecom      # 企业微信
uv install --extra weixin      # 微信
uv install --extra matrix      # Matrix

# （可选）安装开发依赖
uv install --extra dev
```

### 2. 初始配置

```bash
# 快速配置 — 创建 ~/.nanobot/config.json
uv run nanobot onboard

# 交互式向导 — 选择服务商、模型、渠道
uv run nanobot onboard --wizard
```

或直接设置环境变量：
```bash
export NANOBOT_PROVIDERS__ANTHROPIC__API_KEY="sk-ant-..."
export NANOBOT_PROVIDERS__OPENAI__API_KEY="sk-..."
export NANOBOT_AGENTS__DEFAULTS__MODEL="anthropic/claude-sonnet-4-20250514"
```

### 3. 运行

```bash
# 交互式 CLI（REPL 模式）
uv run nanobot

# 启动网关（所有渠道 + 定时任务 + 心跳）
uv run nanobot gateway

# 自定义端口启动网关
uv run nanobot gateway --port 8080

# 查看状态
uv run nanobot status

# 渠道认证
uv run nanobot channels login telegram
```

### 4. 构建与运行 WhatsApp 桥接（可选）

```bash
cd bridge
npm install
npm run build
npm start      # 或: npm run dev
```

### 5. Docker 部署

```bash
# 构建并运行网关
docker-compose up -d nanobot-gateway

# 在容器中运行 CLI
docker-compose run --rm nanobot-cli

# 或手动构建
docker build -t nanobot .
docker run -v ~/.nanobot:/root/.nanobot -p 18790:18790 nanobot gateway
```

### 6. 开发工作流

```bash
# 运行测试
pytest

# 运行测试（含覆盖率）
pytest --cov=nanobot

# 代码检查
ruff check .

# 代码格式化
ruff format .
```
