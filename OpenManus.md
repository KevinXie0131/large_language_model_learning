# CLAUDE.md

---

# English Version

## Project Overview

**OpenManus** is an open-source AI agent framework for building general-purpose autonomous AI agents. It supports multiple LLM providers, browser automation, code execution, web search, multi-agent orchestration, sandboxed execution, and MCP (Model Context Protocol) integration.

Repository: [https://github.com/mannaandpoem/OpenManus](https://github.com/mannaandpoem/OpenManus)

---

## Tech Stack, Frameworks & Versions

### Core

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.12+ | Runtime |
| pydantic | ~2.10.6 | Data validation & settings |
| pydantic_core | ~2.27.2 | Pydantic internals |
| openai | ~1.66.3 | LLM API client (OpenAI, Azure, compatible APIs) |
| tiktoken | ~0.9.0 | Token counting |
| tenacity | ~9.0.0 | Retry with exponential backoff |
| loguru | ~0.7.3 | Structured logging |
| fastapi | ~0.115.11 | Async web framework |
| uvicorn | ~0.34.0 | ASGI server |
| httpx | >=0.27.0 | Async HTTP client |
| tomli | >=2.0.0 | TOML config parsing |
| pyyaml | ~6.0.2 | YAML support |

### LLM Providers

| Provider | Package / Integration | Notes |
|----------|----------------------|-------|
| OpenAI | openai ~1.66.3 (AsyncOpenAI) | GPT-4o, GPT-4o-mini, o1, o3, o4-mini, GPT-5 |
| Azure OpenAI | openai ~1.66.3 (AsyncAzureOpenAI) | Azure-hosted OpenAI models |
| Anthropic Claude | Via OpenAI-compatible endpoint | Claude 3 Opus/Sonnet/Haiku |
| AWS Bedrock | boto3 ~1.37.18 (BedrockClient) | Bedrock-hosted models |
| Ollama | Via OpenAI-compatible endpoint | Local models (Llama, etc.) |
| JiekouAI | Via OpenAI-compatible endpoint | Third-party proxy |

### Browser Automation

| Package | Version | Purpose |
|---------|---------|---------|
| playwright | ~1.51.0 | Browser control engine |
| browser-use | ~0.1.40 | High-level browser automation |
| browsergym | ~0.13.3 | Browser environment for agents |

### Web & Search

| Package | Version | Purpose |
|---------|---------|---------|
| googlesearch-python | ~1.3.0 | Google search |
| duckduckgo_search | ~7.5.3 | DuckDuckGo search |
| baidusearch | ~1.0.3 | Baidu search |
| crawl4ai | ~0.6.3 | Web crawling |
| beautifulsoup4 | ~4.13.3 | HTML parsing |
| requests | ~2.32.3 | HTTP client |
| html2text | ~2024.2.26 | HTML to Markdown |

### Sandbox & Container

| Package | Version | Purpose |
|---------|---------|---------|
| docker | ~7.1.0 | Docker container management |

### Data & ML

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | latest | Numerical computing |
| datasets | ~3.4.1 | HuggingFace datasets |
| huggingface-hub | ~0.29.2 | HuggingFace model hub |
| pillow | >=10.4 | Image processing |

### MCP (Model Context Protocol)

| Package | Version | Purpose |
|---------|---------|---------|
| mcp | ~1.5.0 | MCP protocol SDK |

### Dev & Testing

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | ~8.3.5 | Test framework |
| pytest-asyncio | ~0.25.3 | Async test support |
| black | (pre-commit) | Code formatter |
| isort | (pre-commit) | Import sorter |
| autoflake | (pre-commit) | Unused import removal |

### Other

| Package | Version | Purpose |
|---------|---------|---------|
| gymnasium | ~1.1.1 | RL environment interface |
| unidiff | ~0.7.5 | Unified diff parsing |
| aiofiles | ~24.1.0 | Async file I/O |
| colorama | ~0.4.6 | Terminal color output |
| setuptools | ~75.8.0 | Package utilities |

---

## Features & Functionality

### Agent System
- **Manus Agent** — General-purpose agent with Python execution, browser automation, file editing, human interaction, and MCP tool integration.
- **SWE Agent** — Software engineering agent with Bash, file editing (StrReplaceEditor), and code analysis tools.
- **Browser Agent** — Web browsing agent with screenshot-aware context, page interaction (click, scroll, input, extract), and navigation.
- **MCP Agent** — Connects to MCP servers (SSE or stdio) for dynamic tool discovery and execution.
- **DataAnalysis Agent** — Specialized for data analysis with Python execution and visualization tools.
- **SandboxManus Agent** — Containerized variant of Manus running inside Docker/Daytona sandboxes with isolated shell, browser, and file tools.

### Tool System
- **PythonExecute** — Execute arbitrary Python code with timeout and multiprocess safety.
- **Bash** — Run shell commands with timeout and interactive mode support.
- **StrReplaceEditor** — File operations: view, create, string replace, insert, and undo edits.
- **BrowserUseTool** — Web automation: navigate, click, type, scroll, extract content, switch tabs.
- **WebSearch** — Multi-engine search with fallback (Google, DuckDuckGo, Bing, Baidu).
- **AskHuman** — Request human input during agent execution.
- **PlanningTool** — Plan creation, update, and step tracking for multi-step workflows.
- **CreateChatCompletion** — Nested LLM calls within tool execution.
- **MCPClientTool** — Proxy tool dynamically created from MCP server tool schemas.
- **Terminate** — Signal agent completion (success/failure).
- **Visualization tools** — VisualizationPrepare, DataVisualization for data analysis.

### Multi-Agent Orchestration
- **PlanningFlow** — Plan-driven execution: create a plan, assign steps to specialized agents, track progress with status markers ([✓] completed, [→] in progress, [!] blocked, [ ] not started).
- **FlowFactory** — Factory pattern for creating flows with single or multiple agents.

### MCP Integration
- **MCP Client** — Connect to external MCP servers to discover and use tools dynamically.
- **MCP Server** — Expose OpenManus tools (Bash, Browser, FileEditor, Terminate) as an MCP server for other systems.
- Supports both SSE and stdio transport.

### Sandboxed Execution
- Docker container isolation with configurable resource limits (memory, CPU).
- Sandbox-specific tools: SandboxShellTool, SandboxBrowserTool, SandboxFilesTool, SandboxVisionTool.
- Optional Daytona sandbox integration.

### LLM Features
- Singleton LLM instances per configuration name.
- Token counting with tiktoken (text + image tokens).
- Cumulative token usage tracking with configurable `max_input_tokens` limit.
- Automatic retry with exponential backoff (up to 6 attempts).
- Streaming and non-streaming response modes.
- Multimodal (image) support for compatible models.
- Reasoning model support (o1, o3, o3-mini, o4-mini, GPT-5).

### Robustness
- Loop/stuck detection — detects repeated identical responses and injects escape prompts.
- Token limit management — raises `TokenLimitExceeded` to prevent runaway costs.
- Graceful error handling with structured logging (loguru, per-session log files).

---

## System Design

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                       Entry Points                           │
│  main.py | run_flow.py | run_mcp.py | sandbox_main.py       │
│                      run_mcp_server.py                       │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
       ┌───────▼────────┐             ┌────────▼────────┐
       │   Agent Layer   │             │   Flow Layer    │
       │                 │             │                 │
       │  BaseAgent      │◄────────────│  BaseFlow       │
       │  ├─ ReActAgent  │             │  └─ Planning    │
       │  │  └─ ToolCall │             │     Flow        │
       │  │     ├─ Manus │             │                 │
       │  │     ├─ SWE   │             │  FlowFactory    │
       │  │     ├─ MCP   │             └─────────────────┘
       │  │     ├─Browser│
       │  │     └─DataAn.│
       │  └─ Sandbox     │
       └───────┬─────────┘
               │
       ┌───────▼─────────┐
       │   Tool Layer     │
       │                  │
       │  ToolCollection  │
       │  ├─ PythonExec   │
       │  ├─ Bash         │
       │  ├─ BrowserUse   │
       │  ├─ StrReplace   │
       │  ├─ WebSearch    │
       │  ├─ Planning     │
       │  ├─ AskHuman     │
       │  ├─ Terminate    │
       │  └─ MCPClient    │
       └───────┬──────────┘
               │
       ┌───────▼─────────┐       ┌─────────────────┐
       │   LLM Layer      │       │   MCP Layer      │
       │                  │       │                  │
       │  LLM (Singleton) │       │  MCPClients      │
       │  ├─ AsyncOpenAI  │       │  MCPServer       │
       │  ├─ AsyncAzure   │       │  MCPClientTool   │
       │  └─ Bedrock      │       └──────────────────┘
       │  TokenCounter    │
       └──────────────────┘
               │
       ┌───────▼─────────┐       ┌─────────────────┐
       │   Config Layer   │       │  Sandbox Layer   │
       │                  │       │                  │
       │  AppConfig       │       │  DockerSandbox   │
       │  LLMSettings     │       │  SandboxClient   │
       │  BrowserSettings │       │  SandboxTools    │
       │  SearchSettings  │       └──────────────────┘
       │  SandboxSettings │
       │  MCPSettings     │
       └──────────────────┘
```

### Class Hierarchy

```
BaseAgent (abstract)
├── ReActAgent (abstract) — think() → act() loop
│   └── ToolCallAgent (concrete) — LLM function calling
│       ├── Manus — General-purpose, MCP-enabled
│       │   └── SandboxManus — Containerized execution
│       ├── SWEAgent — Software engineering
│       ├── BrowserAgent — Web browsing
│       ├── MCPAgent — MCP protocol integration
│       └── DataAnalysis — Data analysis & visualization
└── SandboxAgent — Base for sandboxed agents

BaseFlow (abstract)
└── PlanningFlow — Plan-driven multi-agent orchestration

BaseTool (abstract)
├── PythonExecute, Bash, StrReplaceEditor
├── BrowserUseTool, WebSearch
├── AskHuman, Terminate, PlanningTool
├── CreateChatCompletion
└── MCPClientTool (dynamic proxy)

ToolCollection — Tool aggregation & dispatch
└── MCPClients — MCP-specific tool collection with server connections
```

### Core Execution Flow

1. **Single Agent (`BaseAgent.run`)**:
   - Transition to RUNNING state.
   - Add user request to memory.
   - Loop: `step()` → check `is_stuck()` → handle if stuck → repeat until `max_steps` or FINISHED.
   - Cleanup and return to IDLE.

2. **ReAct Step (`ReActAgent.step`)**:
   - `think()` — Decide next action via LLM.
   - `act()` — Execute the decided action.

3. **Tool Calling (`ToolCallAgent`)**:
   - `think()` — Send memory + tools to LLM → extract `tool_calls`.
   - `act()` — For each tool call: parse arguments → `ToolCollection.execute()` → add result to memory → check for special tools (Terminate).

4. **Planning Flow (`PlanningFlow.execute`)**:
   - Create plan from user input using PlanningTool.
   - For each step: select executor agent → run agent on step → mark step status.
   - Adapt plan based on results.

### Design Patterns

| Pattern | Usage |
|---------|-------|
| **Singleton** | `LLM` (per config_name), `Config` (thread-safe) |
| **Factory** | `FlowFactory.create_flow()`, `Manus.create()` async factory |
| **Strategy** | `ToolChoice` (AUTO/REQUIRED/NONE), agent selection per step |
| **Template Method** | `BaseAgent.run()` → `step()`, `ReActAgent.step()` → `think()`/`act()` |
| **Composite** | `ToolCollection` aggregates tools, `BaseFlow` aggregates agents |
| **Context Manager** | `state_context()` for safe state transitions, `AsyncExitStack` for MCP cleanup |
| **Chain of Responsibility** | ReAct loop: think → act → observe → repeat |

### Configuration Architecture

- **TOML-based** config files (`config/config.toml`).
- **Hierarchical settings**: `AppConfig` → `LLMSettings`, `BrowserSettings`, `SearchSettings`, `SandboxSettings`, `MCPSettings`, `DaytonaSettings`, `RunflowSettings`.
- Supports multiple named LLM configs (default + overrides like `llm.vision`).
- MCP servers configured via JSON (`config/mcp.json`).

---

## How to Build and Run Locally

### Prerequisites

- Python 3.12+
- Git
- (Optional) Docker — for sandboxed execution
- (Optional) Conda or uv — for virtual environment management

### 1. Clone the Repository

```bash
git clone https://github.com/mannaandpoem/OpenManus.git
cd OpenManus
```

### 2. Create Virtual Environment

**Option A: Conda**
```bash
conda create -n open_manus python=3.12
conda activate open_manus
```

**Option B: uv**
```bash
uv venv --python 3.12
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

**Option C: venv**
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Browser for Automation

```bash
playwright install
```

### 5. Configure

```bash
# Copy and edit the LLM configuration
cp config/config.example.toml config/config.toml
```

Edit `config/config.toml` — set your LLM provider, API key, model, and other options. Examples for each provider are commented in the file. Key settings:

```toml
[llm]
model = "claude-3-7-sonnet-20250219"
base_url = "https://api.anthropic.com/v1/"
api_key = "YOUR_API_KEY"
max_tokens = 8192
temperature = 0.0
```

For MCP servers:
```bash
cp config/mcp.example.json config/mcp.json
```

### 6. Run

| Command | Description |
|---------|-------------|
| `python main.py` | Interactive mode — general-purpose Manus agent |
| `python main.py --prompt "Your task"` | Single prompt mode |
| `python run_flow.py` | Multi-agent planning flow |
| `python run_mcp.py` | MCP agent (stdio by default) |
| `python run_mcp.py -c sse --server-url http://localhost:8000/sse` | MCP agent via SSE |
| `python run_mcp.py -i` | MCP agent interactive mode |
| `python sandbox_main.py` | Sandboxed agent (requires Docker) |
| `python run_mcp_server.py` | Start MCP server (exposes tools) |

### 7. Testing

```bash
pytest tests/
```

### 8. Pre-commit (Linting & Formatting)

```bash
pre-commit run --all-files
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `app/agent/` | Agent implementations (Manus, MCP, Browser, SWE, DataAnalysis, Sandbox) |
| `app/tool/` | Tool implementations (Python exec, browser, bash, search, file editing, etc.) |
| `app/flow/` | Multi-agent workflow orchestration |
| `app/prompt/` | System prompts for each agent type |
| `app/mcp/` | MCP server implementation |
| `app/sandbox/` | Sandboxed execution utilities (Docker) |
| `config/` | Configuration files (TOML + JSON) |
| `workspace/` | Agent working directory (created at runtime) |
| `logs/` | Log output (created at runtime) |

---
---

# Chinese Version / 中文版

## 项目概述

**OpenManus** 是一个开源的 AI 智能体框架，用于构建通用自主 AI 智能体。它支持多种 LLM 提供商、浏览器自动化、代码执行、网络搜索、多智能体协作、沙箱执行以及 MCP（Model Context Protocol）集成。

仓库地址：[https://github.com/mannaandpoem/OpenManus](https://github.com/mannaandpoem/OpenManus)

---

## 技术栈、框架与版本

### 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 运行时 |
| pydantic | ~2.10.6 | 数据验证与配置管理 |
| pydantic_core | ~2.27.2 | Pydantic 内部依赖 |
| openai | ~1.66.3 | LLM API 客户端（OpenAI、Azure、兼容 API） |
| tiktoken | ~0.9.0 | Token 计数 |
| tenacity | ~9.0.0 | 指数退避重试机制 |
| loguru | ~0.7.3 | 结构化日志 |
| fastapi | ~0.115.11 | 异步 Web 框架 |
| uvicorn | ~0.34.0 | ASGI 服务器 |
| httpx | >=0.27.0 | 异步 HTTP 客户端 |
| tomli | >=2.0.0 | TOML 配置解析 |
| pyyaml | ~6.0.2 | YAML 支持 |

### LLM 提供商

| 提供商 | 包 / 集成方式 | 说明 |
|--------|-------------|------|
| OpenAI | openai ~1.66.3 (AsyncOpenAI) | GPT-4o、GPT-4o-mini、o1、o3、o4-mini、GPT-5 |
| Azure OpenAI | openai ~1.66.3 (AsyncAzureOpenAI) | Azure 托管的 OpenAI 模型 |
| Anthropic Claude | 通过 OpenAI 兼容端点 | Claude 3 Opus/Sonnet/Haiku |
| AWS Bedrock | boto3 ~1.37.18 (BedrockClient) | Bedrock 托管模型 |
| Ollama | 通过 OpenAI 兼容端点 | 本地模型（Llama 等） |
| JiekouAI | 通过 OpenAI 兼容端点 | 第三方代理 |

### 浏览器自动化

| 包名 | 版本 | 用途 |
|------|------|------|
| playwright | ~1.51.0 | 浏览器控制引擎 |
| browser-use | ~0.1.40 | 高级浏览器自动化 |
| browsergym | ~0.13.3 | 智能体浏览器环境 |

### 网络与搜索

| 包名 | 版本 | 用途 |
|------|------|------|
| googlesearch-python | ~1.3.0 | Google 搜索 |
| duckduckgo_search | ~7.5.3 | DuckDuckGo 搜索 |
| baidusearch | ~1.0.3 | 百度搜索 |
| crawl4ai | ~0.6.3 | 网页爬取 |
| beautifulsoup4 | ~4.13.3 | HTML 解析 |
| requests | ~2.32.3 | HTTP 客户端 |
| html2text | ~2024.2.26 | HTML 转 Markdown |

### 沙箱与容器

| 包名 | 版本 | 用途 |
|------|------|------|
| docker | ~7.1.0 | Docker 容器管理 |

### 数据与机器学习

| 包名 | 版本 | 用途 |
|------|------|------|
| numpy | 最新版 | 数值计算 |
| datasets | ~3.4.1 | HuggingFace 数据集 |
| huggingface-hub | ~0.29.2 | HuggingFace 模型中心 |
| pillow | >=10.4 | 图像处理 |

### MCP（Model Context Protocol）

| 包名 | 版本 | 用途 |
|------|------|------|
| mcp | ~1.5.0 | MCP 协议 SDK |

### 开发与测试

| 包名 | 版本 | 用途 |
|------|------|------|
| pytest | ~8.3.5 | 测试框架 |
| pytest-asyncio | ~0.25.3 | 异步测试支持 |
| black | (pre-commit) | 代码格式化 |
| isort | (pre-commit) | Import 排序 |
| autoflake | (pre-commit) | 移除未使用的 import |

### 其他

| 包名 | 版本 | 用途 |
|------|------|------|
| gymnasium | ~1.1.1 | 强化学习环境接口 |
| unidiff | ~0.7.5 | Unified diff 解析 |
| aiofiles | ~24.1.0 | 异步文件 I/O |
| colorama | ~0.4.6 | 终端彩色输出 |
| setuptools | ~75.8.0 | 包管理工具 |

---

## 功能特性

### 智能体系统
- **Manus 智能体** — 通用智能体，集成 Python 执行、浏览器自动化、文件编辑、人机交互和 MCP 工具。
- **SWE 智能体** — 软件工程智能体，配备 Bash、文件编辑（StrReplaceEditor）和代码分析工具。
- **Browser 智能体** — 网页浏览智能体，支持截图感知上下文、页面交互（点击、滚动、输入、内容提取）和导航。
- **MCP 智能体** — 连接 MCP 服务器（SSE 或 stdio），实现动态工具发现和执行。
- **DataAnalysis 智能体** — 数据分析专用智能体，支持 Python 执行和可视化工具。
- **SandboxManus 智能体** — Manus 的容器化版本，在 Docker/Daytona 沙箱中运行，拥有隔离的 Shell、浏览器和文件工具。

### 工具系统
- **PythonExecute** — 执行任意 Python 代码，支持超时和多进程安全。
- **Bash** — 运行 Shell 命令，支持超时和交互模式。
- **StrReplaceEditor** — 文件操作：查看、创建、字符串替换、插入、撤销编辑。
- **BrowserUseTool** — 网页自动化：导航、点击、输入、滚动、提取内容、切换标签页。
- **WebSearch** — 多引擎搜索，支持故障转移（Google、DuckDuckGo、Bing、百度）。
- **AskHuman** — 在智能体执行过程中请求人工输入。
- **PlanningTool** — 计划创建、更新和步骤跟踪。
- **CreateChatCompletion** — 在工具执行中嵌套调用 LLM。
- **MCPClientTool** — 从 MCP 服务器工具 Schema 动态创建的代理工具。
- **Terminate** — 标记智能体完成（成功/失败）。
- **可视化工具** — VisualizationPrepare、DataVisualization，用于数据分析。

### 多智能体编排
- **PlanningFlow** — 计划驱动执行：创建计划，将步骤分配给专业智能体，通过状态标记跟踪进度（[✓] 已完成、[→] 进行中、[!] 阻塞、[ ] 未开始）。
- **FlowFactory** — 工厂模式，支持单个或多个智能体创建工作流。

### MCP 集成
- **MCP 客户端** — 连接外部 MCP 服务器，动态发现和使用工具。
- **MCP 服务器** — 将 OpenManus 工具（Bash、Browser、FileEditor、Terminate）作为 MCP 服务器暴露给其他系统。
- 支持 SSE 和 stdio 两种传输方式。

### 沙箱执行
- Docker 容器隔离，可配置资源限制（内存、CPU）。
- 沙箱专用工具：SandboxShellTool、SandboxBrowserTool、SandboxFilesTool、SandboxVisionTool。
- 可选 Daytona 沙箱集成。

### LLM 特性
- 按配置名的单例 LLM 实例。
- 基于 tiktoken 的 Token 计数（文本 + 图像 Token）。
- 累计 Token 使用量跟踪，可配置 `max_input_tokens` 上限。
- 自动指数退避重试（最多 6 次）。
- 流式和非流式响应模式。
- 多模态（图像）支持。
- 推理模型支持（o1、o3、o3-mini、o4-mini、GPT-5）。

### 鲁棒性
- 循环/卡死检测 — 检测重复响应并注入逃脱提示。
- Token 限额管理 — 抛出 `TokenLimitExceeded` 防止成本失控。
- 结构化日志（loguru，按会话生成日志文件）的优雅错误处理。

---

## 系统设计

### 架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                         入口点                                │
│  main.py | run_flow.py | run_mcp.py | sandbox_main.py        │
│                      run_mcp_server.py                        │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
       ┌───────▼────────┐             ┌────────▼────────┐
       │    智能体层      │             │     工作流层     │
       │                 │             │                 │
       │  BaseAgent      │◄────────────│  BaseFlow       │
       │  ├─ ReActAgent  │             │  └─ Planning    │
       │  │  └─ ToolCall │             │     Flow        │
       │  │     ├─ Manus │             │                 │
       │  │     ├─ SWE   │             │  FlowFactory    │
       │  │     ├─ MCP   │             └─────────────────┘
       │  │     ├─Browser│
       │  │     └─DataAn.│
       │  └─ Sandbox     │
       └───────┬─────────┘
               │
       ┌───────▼─────────┐
       │     工具层        │
       │                  │
       │  ToolCollection  │
       │  ├─ PythonExec   │
       │  ├─ Bash         │
       │  ├─ BrowserUse   │
       │  ├─ StrReplace   │
       │  ├─ WebSearch    │
       │  ├─ Planning     │
       │  ├─ AskHuman     │
       │  ├─ Terminate    │
       │  └─ MCPClient    │
       └───────┬──────────┘
               │
       ┌───────▼─────────┐       ┌─────────────────┐
       │     LLM 层       │       │     MCP 层       │
       │                  │       │                  │
       │  LLM (单例)      │       │  MCPClients      │
       │  ├─ AsyncOpenAI  │       │  MCPServer       │
       │  ├─ AsyncAzure   │       │  MCPClientTool   │
       │  └─ Bedrock      │       └──────────────────┘
       │  TokenCounter    │
       └──────────────────┘
               │
       ┌───────▼─────────┐       ┌─────────────────┐
       │     配置层        │       │     沙箱层       │
       │                  │       │                  │
       │  AppConfig       │       │  DockerSandbox   │
       │  LLMSettings     │       │  SandboxClient   │
       │  BrowserSettings │       │  SandboxTools    │
       │  SearchSettings  │       └──────────────────┘
       │  SandboxSettings │
       │  MCPSettings     │
       └──────────────────┘
```

### 类继承关系

```
BaseAgent（抽象类）
├── ReActAgent（抽象类）— think() → act() 循环
│   └── ToolCallAgent（具体类）— LLM 函数调用
│       ├── Manus — 通用智能体，支持 MCP
│       │   └── SandboxManus — 容器化执行
│       ├── SWEAgent — 软件工程
│       ├── BrowserAgent — 网页浏览
│       ├── MCPAgent — MCP 协议集成
│       └── DataAnalysis — 数据分析与可视化
└── SandboxAgent — 沙箱智能体基类

BaseFlow（抽象类）
└── PlanningFlow — 计划驱动的多智能体编排

BaseTool（抽象类）
├── PythonExecute、Bash、StrReplaceEditor
├── BrowserUseTool、WebSearch
├── AskHuman、Terminate、PlanningTool
├── CreateChatCompletion
└── MCPClientTool（动态代理）

ToolCollection — 工具聚合与分发
└── MCPClients — MCP 专用工具集合，管理服务器连接
```

### 核心执行流程

1. **单智能体执行（`BaseAgent.run`）**：
   - 切换到 RUNNING 状态。
   - 将用户请求添加到记忆中。
   - 循环：`step()` → 检查 `is_stuck()` → 如果卡住则处理 → 重复直到 `max_steps` 或 FINISHED。
   - 清理资源并返回 IDLE 状态。

2. **ReAct 步骤（`ReActAgent.step`）**：
   - `think()` — 通过 LLM 决定下一步动作。
   - `act()` — 执行决定的动作。

3. **工具调用（`ToolCallAgent`）**：
   - `think()` — 将记忆 + 工具发送给 LLM → 提取 `tool_calls`。
   - `act()` — 对每个工具调用：解析参数 → `ToolCollection.execute()` → 将结果加入记忆 → 检查特殊工具（Terminate）。

4. **计划工作流（`PlanningFlow.execute`）**：
   - 使用 PlanningTool 从用户输入创建计划。
   - 对每个步骤：选择执行智能体 → 运行智能体执行步骤 → 标记步骤状态。
   - 根据结果调整计划。

### 设计模式

| 模式 | 应用场景 |
|------|---------|
| **单例模式** | `LLM`（按 config_name）、`Config`（线程安全） |
| **工厂模式** | `FlowFactory.create_flow()`、`Manus.create()` 异步工厂 |
| **策略模式** | `ToolChoice`（AUTO/REQUIRED/NONE）、按步骤类型选择智能体 |
| **模板方法** | `BaseAgent.run()` → `step()`、`ReActAgent.step()` → `think()`/`act()` |
| **组合模式** | `ToolCollection` 聚合工具、`BaseFlow` 聚合智能体 |
| **上下文管理器** | `state_context()` 安全状态转换、`AsyncExitStack` MCP 资源清理 |
| **责任链** | ReAct 循环：思考 → 执行 → 观察 → 重复 |

### 配置架构

- 基于 **TOML** 的配置文件（`config/config.toml`）。
- **层级配置**：`AppConfig` → `LLMSettings`、`BrowserSettings`、`SearchSettings`、`SandboxSettings`、`MCPSettings`、`DaytonaSettings`、`RunflowSettings`。
- 支持多个命名 LLM 配置（default + 覆盖配置如 `llm.vision`）。
- MCP 服务器通过 JSON 配置（`config/mcp.json`）。

---

## 本地构建与运行

### 前置要求

- Python 3.12+
- Git
- （可选）Docker — 用于沙箱执行
- （可选）Conda 或 uv — 用于虚拟环境管理

### 1. 克隆仓库

```bash
git clone https://github.com/mannaandpoem/OpenManus.git
cd OpenManus
```

### 2. 创建虚拟环境

**方式 A：Conda**
```bash
conda create -n open_manus python=3.12
conda activate open_manus
```

**方式 B：uv**
```bash
uv venv --python 3.12
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

**方式 C：venv**
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. （可选）安装浏览器自动化

```bash
playwright install
```

### 5. 配置

```bash
# 复制并编辑 LLM 配置
cp config/config.example.toml config/config.toml
```

编辑 `config/config.toml` — 设置你的 LLM 提供商、API Key、模型和其他选项。文件中包含各提供商的注释示例。关键配置：

```toml
[llm]
model = "claude-3-7-sonnet-20250219"
base_url = "https://api.anthropic.com/v1/"
api_key = "YOUR_API_KEY"
max_tokens = 8192
temperature = 0.0
```

配置 MCP 服务器：
```bash
cp config/mcp.example.json config/mcp.json
```

### 6. 运行

| 命令 | 说明 |
|------|------|
| `python main.py` | 交互模式 — 通用 Manus 智能体 |
| `python main.py --prompt "你的任务"` | 单次提示模式 |
| `python run_flow.py` | 多智能体计划工作流 |
| `python run_mcp.py` | MCP 智能体（默认 stdio） |
| `python run_mcp.py -c sse --server-url http://localhost:8000/sse` | MCP 智能体（SSE 模式） |
| `python run_mcp.py -i` | MCP 智能体交互模式 |
| `python sandbox_main.py` | 沙箱智能体（需要 Docker） |
| `python run_mcp_server.py` | 启动 MCP 服务器（暴露工具） |

### 7. 测试

```bash
pytest tests/
```

### 8. 预提交检查（代码检查与格式化）

```bash
pre-commit run --all-files
```

### 关键目录

| 目录 | 用途 |
|------|------|
| `app/agent/` | 智能体实现（Manus、MCP、Browser、SWE、DataAnalysis、Sandbox） |
| `app/tool/` | 工具实现（Python 执行、浏览器、Bash、搜索、文件编辑等） |
| `app/flow/` | 多智能体工作流编排 |
| `app/prompt/` | 各智能体的系统提示词 |
| `app/mcp/` | MCP 服务器实现 |
| `app/sandbox/` | 沙箱执行工具（Docker） |
| `config/` | 配置文件（TOML + JSON） |
| `workspace/` | 智能体工作目录（运行时创建） |
| `logs/` | 日志输出（运行时创建） |
