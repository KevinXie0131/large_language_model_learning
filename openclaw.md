AGENTS.md

---

# OpenClaw Project Documentation

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | TypeScript (ESM, strict typing) | 5.9+ |
| Runtime | Node.js | 22.16+ (24 recommended) |
| Alternative Runtime | Bun | supported for TS execution/dev |
| Package Manager | pnpm | 10.32+ |
| Bundler | tsdown | 0.21+ |
| HTTP Framework | Hono | 4.12+ |
| HTTP Server (legacy) | Express | 5.2+ |
| WebSocket | ws | 8.20+ |
| Test Framework | Vitest (V8 coverage) | 4.1+ |
| Linter | Oxlint | 1.56+ |
| Formatter | Oxfmt | 0.41+ |
| Type Checker | TypeScript / tsgo | 5.9+ / 7.0-preview |
| UI Framework | Lit (Web Components) | 3.3+ |
| UI Build Tool | Vite | 8.0+ |
| Browser Automation | Playwright | 1.58+ |
| Schema Validation | Zod / @sinclair/typebox / ajv | 4.3+ / 0.34+ / 8.18+ |
| Image Processing | sharp | 0.34+ |
| Embedded DB | sqlite-vec | 0.1.7 |
| PDF Parsing | pdfjs-dist | 5.5+ |
| macOS App | SwiftUI (Observation framework) | Swift 5.9+ |
| iOS App | SwiftUI + XcodeGen | Swift 5.9+ |
| Android App | Kotlin + Jetpack Compose | Gradle/KTS |
| Agent Runtime | Pi agent (mariozechner/pi-*) | 0.61+ |
| MCP Protocol | @modelcontextprotocol/sdk | 1.27+ |
| ACP Protocol | @agentclientprotocol/sdk | 0.16+ |

## Key Dependencies

- **AI/LLM**: Anthropic (Claude), OpenAI, Google, DeepSeek, Groq, Mistral, Ollama, and 70+ providers via plugins
- **Messaging Channels**: Baileys (WhatsApp), grammY types (Telegram), discord.js (Discord), Bolt (Slack), signal-cli (Signal), @line/bot-sdk (LINE), matrix-js-sdk (Matrix)
- **Voice/Speech**: ElevenLabs, Deepgram, node-edge-tts (system TTS fallback)
- **CLI**: Commander 14+, @clack/prompts, chalk 5+, cli-highlight
- **Networking**: undici 7+, gaxios, chokidar 5+ (file watching)
- **Configuration**: dotenv, yaml, json5
- **Utilities**: uuid, croner (cron), jszip, tar, markdown-it, linkedom, qrcode-terminal

## Features and Functionality

### Core Platform

- **Local-first Gateway** -- Single WebSocket control plane (`ws://127.0.0.1:18789`) managing sessions, channels, tools, events, webhooks, and cron jobs
- **CLI Surface** -- Full CLI (`openclaw`) with commands for gateway, agent, send, onboarding, doctor, config, plugins, devices, nodes, cron, skills, etc. (308+ command files)
- **Pi Agent Runtime** -- RPC-mode agent with tool streaming, block streaming, thinking levels, and multi-session support
- **Session Model** -- `main` session for direct chats, group isolation, activation modes, queue modes, reply-back, and session pruning
- **Control UI** -- Web-based control panel served from the Gateway (Lit + Vite), with chat, config, and monitoring
- **WebChat** -- Browser-based chat interface embedded in the Gateway

### Multi-Channel Messaging (22+ channels)

- **Core channels**: WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, iMessage, BlueBubbles, IRC, WebChat
- **Extension channels**: Microsoft Teams, Matrix, Feishu, LINE, Mattermost, Nextcloud Talk, Nostr, Synology Chat, Tlon, Twitch, Zalo, Zalo Personal
- **Features**: DM pairing/allowlisting, group routing, mention gating, reply tags, per-channel chunking

### AI Provider Plugins (80+ extensions)

- **Major providers**: Anthropic, OpenAI, Google, DeepSeek, Groq, Mistral, Ollama, Hugging Face, Together, Perplexity, xAI, NVIDIA
- **Regional providers**: BytePlus, VolcEngine, Qianfan, MiniMax, Moonshot, Xiaomi, ModelStudio, Kimi Coding
- **Gateway proxies**: Cloudflare AI Gateway, Vercel AI Gateway, OpenRouter, Copilot Proxy, GitHub Copilot
- **Other**: Amazon Bedrock, Anthropic Vertex, Venice, Chutes, sglang, vLLM

### Companion Apps

- **macOS App** -- SwiftUI menu bar app with gateway control, Voice Wake/PTT, Talk Mode overlay, WebChat, debug tools, remote gateway control
- **iOS App** -- SwiftUI node with Canvas, Voice Wake, Talk Mode, camera, screen recording, Bonjour + device pairing, Watch app
- **Android App** -- Kotlin/Compose node with Connect/Chat/Voice tabs, Canvas, camera/screen recording, device commands (notifications/location/SMS/photos/contacts/calendar/motion)

### Tools and Automation

- **Browser Control** -- Managed Chrome/Chromium with CDP, snapshots, actions, uploads, profiles
- **Live Canvas** -- Agent-driven visual workspace with A2UI (Angular/Lit renderers)
- **Nodes** -- Camera snap/clip, screen record, location.get, notifications, system.run/system.notify (macOS)
- **Cron and Wakeups** -- Scheduled agent tasks and webhook-triggered actions
- **Skills Platform** -- Bundled, managed, and workspace skills (ClawHub registry)
- **Media Pipeline** -- Images, audio, video processing; transcription hooks; size caps; temp file lifecycle
- **Voice** -- Voice Wake (wake words on macOS/iOS), Talk Mode (continuous voice on Android), TTS (ElevenLabs + system fallback)
- **Image Generation** -- Via fal and other provider plugins
- **Web Search** -- Brave, DuckDuckGo, Exa, Tavily, Firecrawl, Perplexity
- **Memory** -- memory-core + memory-lancedb extensions for persistent agent memory
- **Agent-to-Agent** -- sessions_list, sessions_history, sessions_send for cross-session coordination

### Security

- **DM Pairing** -- Unknown senders receive a pairing code; allowlist-based access control
- **Sandbox Mode** -- Per-session Docker sandboxes for non-main sessions (groups/channels)
- **Auth Modes** -- Token/password auth, Tailscale identity headers, browser hardening, SSRF protection
- **Exec Approval** -- Configurable command approval policies for tool execution

## System Design

```
Messaging Channels (WhatsApp / Telegram / Slack / Discord / Google Chat / Signal / iMessage / ...)
               |
               v
+-------------------------------+
|           Gateway             |
|      (WS control plane)      |  <-- Config: ~/.openclaw/openclaw.json
|   ws://127.0.0.1:18789       |  <-- Credentials: ~/.openclaw/credentials/
|                               |  <-- Sessions: ~/.openclaw/agents/<id>/sessions/
|  +----------+  +-----------+  |
|  | Channels |  |  Routing  |  |
|  +----------+  +-----------+  |
|  +----------+  +-----------+  |
|  | Sessions |  |  Plugins  |  |
|  +----------+  +-----------+  |
|  +----------+  +-----------+  |
|  |  Tools   |  |   Cron    |  |
|  +----------+  +-----------+  |
|  +----------+  +-----------+  |
|  | Webhooks |  | Control UI|  |
|  +----------+  +-----------+  |
+---------------+---------------+
                |
    +-----------+-----------+
    |           |           |
    v           v           v
Pi Agent    CLI Client   Companion Apps
 (RPC)    (openclaw ...)  (macOS/iOS/Android)
```

### Directory Structure

```
openclaw/
  src/                  # Core source code
    cli/                #   CLI wiring and commands
    commands/           #   Command implementations (308+ files)
    gateway/            #   Gateway server, WS control plane, auth, sessions
    channels/           #   Channel routing, contracts, shared channel logic
    routing/            #   Message routing layer
    plugins/            #   Plugin system, contracts, runtime
    plugin-sdk/         #   Plugin SDK (public API surface for extensions)
    agents/             #   Agent runtime, workspace, skills
    media/              #   Media pipeline (images, audio, video)
    media-understanding/#   Media analysis (vision, transcription)
    web-search/         #   Web search provider abstraction
    browser/            #   Browser automation (Playwright/CDP)
    canvas-host/        #   Canvas + A2UI host
    infra/              #   Infrastructure utilities
    security/           #   Security, SSRF, auth primitives
    config/             #   Configuration schema and management
    hooks/              #   Hook system
    interactive/        #   Interactive prompt UI
    tts/                #   Text-to-speech
    cron/               #   Cron job management
    terminal/           #   Terminal UI (tables, palette, progress)
    tui/                #   Terminal UI mode
    process/            #   Process management
    daemon/             #   Daemon (launchd/systemd) management
    sessions/           #   Session persistence
    i18n/               #   Internationalization
    types/              #   Shared type definitions
    utils/              #   Shared utilities
  extensions/           # Plugin packages (80+ extensions)
    anthropic/          #   Anthropic provider
    openai/             #   OpenAI provider
    telegram/           #   Telegram channel
    discord/            #   Discord channel
    slack/              #   Slack channel
    whatsapp/           #   WhatsApp channel
    ...                 #   (70+ more)
  ui/                   # Control UI (Lit + Vite)
  apps/
    macos/              # macOS menu bar app (SwiftUI)
    ios/                # iOS node app (SwiftUI + XcodeGen)
    android/            # Android node app (Kotlin + Compose)
    shared/             # Shared cross-platform code (OpenClawKit)
  docs/                 # Documentation (Mintlify, hosted at docs.openclaw.ai)
  packages/
    clawdbot/           # ClawdBot package
    moltbot/            # MoltBot package
  vendor/
    a2ui/               # A2UI specification + renderers (Angular, Lit)
  scripts/              # Build, release, CI, and utility scripts
  dist/                 # Built output
  skills/               # Bundled skills
```

### Key Architectural Decisions

- **Plugin architecture**: All AI providers and many channels are implemented as plugins under `extensions/`, loaded via the Plugin SDK (`openclaw/plugin-sdk/*`)
- **WebSocket control plane**: The Gateway is the single coordination point; all clients (CLI, apps, UI) connect via WebSocket
- **Local-first**: Gateway runs on the user's machine bound to loopback; remote access via Tailscale Serve/Funnel or SSH tunnels
- **ESM-only**: The entire codebase is ESM TypeScript; dynamic imports use dedicated `*.runtime.ts` boundaries for lazy loading
- **Monorepo**: pnpm workspaces with the core package at root, extensions as workspace packages, and companion apps in `apps/`

## Build and Run Locally

### Prerequisites

- **Node.js** 22.16+ (Node 24 recommended)
- **pnpm** 10.32+ (defined in `packageManager` field)
- **Bun** (optional, for running TypeScript directly)

### Install Dependencies

```bash
git clone https://github.com/openclaw/openclaw.git
cd openclaw
pnpm install
```

### Build

```bash
pnpm ui:build    # Build Control UI (auto-installs UI deps on first run)
pnpm build       # Full TypeScript build -> dist/
```

### Run in Development

```bash
# Run CLI commands directly (TypeScript via tsx, no build needed)
pnpm openclaw ...

# Run the gateway
pnpm openclaw gateway --port 18789 --verbose

# Dev loop with auto-reload on source/config changes
pnpm gateway:watch

# Run the onboarding wizard
pnpm openclaw onboard --install-daemon

# Run the Control UI dev server
pnpm ui:dev
```

### Type Check, Lint, Format

```bash
pnpm tsgo        # TypeScript type checking
pnpm check       # Full check (format + typecheck + lint + SDK checks)
pnpm lint        # Oxlint (type-aware)
pnpm format      # Oxfmt (check mode)
pnpm format:fix  # Oxfmt (auto-fix)
```

### Run Tests

```bash
pnpm test                    # All unit tests (Vitest, parallel)
pnpm test -- <path-or-filter> # Scoped tests
pnpm test:coverage           # With V8 coverage (70% threshold)
pnpm test:e2e                # End-to-end tests
pnpm test:live               # Live tests (requires real API keys: OPENCLAW_LIVE_TEST=1)
pnpm test:gateway            # Gateway-specific tests
pnpm test:ui                 # Control UI tests
```

Low-memory mode:
```bash
OPENCLAW_TEST_PROFILE=low OPENCLAW_TEST_SERIAL_GATEWAY=1 pnpm test
```

### Companion Apps

```bash
# macOS app
pnpm mac:package             # Package (current arch)

# iOS
pnpm ios:gen                 # Generate Xcode project
pnpm ios:build               # Build for simulator
pnpm ios:open                # Open in Xcode

# Android
pnpm android:install         # Build and install on device
pnpm android:run             # Build, install, and launch
pnpm android:test            # Run unit tests
```

### Docker

```bash
# Docker-based tests
pnpm test:docker:all

# Docker build (optimized, skips DTS generation)
pnpm build:docker
```

### Configuration

Config file: `~/.openclaw/openclaw.json`

```json5
{
  agent: {
    model: "anthropic/claude-opus-4-6",
  },
}
```

Full reference: https://docs.openclaw.ai/gateway/configuration

---

# OpenClaw 项目文档 (中文版)

## 技术栈

| 层级 | 技术 | 版本 |
|---|---|---|
| 编程语言 | TypeScript (ESM, 严格类型) | 5.9+ |
| 运行时 | Node.js | 22.16+ (推荐 24) |
| 备选运行时 | Bun | 支持 TS 直接执行/开发 |
| 包管理器 | pnpm | 10.32+ |
| 打包工具 | tsdown | 0.21+ |
| HTTP 框架 | Hono | 4.12+ |
| HTTP 服务器 (旧版) | Express | 5.2+ |
| WebSocket | ws | 8.20+ |
| 测试框架 | Vitest (V8 覆盖率) | 4.1+ |
| 代码检查 | Oxlint | 1.56+ |
| 代码格式化 | Oxfmt | 0.41+ |
| 类型检查 | TypeScript / tsgo | 5.9+ / 7.0-preview |
| UI 框架 | Lit (Web Components) | 3.3+ |
| UI 构建工具 | Vite | 8.0+ |
| 浏览器自动化 | Playwright | 1.58+ |
| 数据验证 | Zod / @sinclair/typebox / ajv | 4.3+ / 0.34+ / 8.18+ |
| 图像处理 | sharp | 0.34+ |
| 嵌入式数据库 | sqlite-vec | 0.1.7 |
| PDF 解析 | pdfjs-dist | 5.5+ |
| macOS 应用 | SwiftUI (Observation 框架) | Swift 5.9+ |
| iOS 应用 | SwiftUI + XcodeGen | Swift 5.9+ |
| Android 应用 | Kotlin + Jetpack Compose | Gradle/KTS |
| 智能体运行时 | Pi agent (mariozechner/pi-*) | 0.61+ |
| MCP 协议 | @modelcontextprotocol/sdk | 1.27+ |
| ACP 协议 | @agentclientprotocol/sdk | 0.16+ |

## 核心依赖

- **AI/LLM**: Anthropic (Claude)、OpenAI、Google、DeepSeek、Groq、Mistral、Ollama 以及 70+ 提供商插件
- **消息通道**: Baileys (WhatsApp)、grammY types (Telegram)、discord.js (Discord)、Bolt (Slack)、signal-cli (Signal)、@line/bot-sdk (LINE)、matrix-js-sdk (Matrix)
- **语音/语音合成**: ElevenLabs、Deepgram、node-edge-tts (系统 TTS 备选方案)
- **CLI**: Commander 14+、@clack/prompts、chalk 5+、cli-highlight
- **网络**: undici 7+、gaxios、chokidar 5+ (文件监听)
- **配置**: dotenv、yaml、json5
- **工具库**: uuid、croner (定时任务)、jszip、tar、markdown-it、linkedom、qrcode-terminal

## 功能特性

### 核心平台

- **本地优先网关** -- 单一 WebSocket 控制平面 (`ws://127.0.0.1:18789`)，管理会话、通道、工具、事件、Webhook 和定时任务
- **CLI 命令行** -- 完整的 CLI 工具 (`openclaw`)，包含 gateway、agent、send、onboard、doctor、config、plugins、devices、nodes、cron、skills 等命令 (308+ 命令文件)
- **Pi 智能体运行时** -- RPC 模式智能体，支持工具流式输出、块流式输出、思考级别和多会话
- **会话模型** -- `main` 会话用于直接聊天，支持群组隔离、激活模式、队列模式、回复转发和会话裁剪
- **控制面板 UI** -- 基于 Web 的控制面板 (Lit + Vite)，从网关直接提供服务，包含聊天、配置和监控功能
- **WebChat** -- 嵌入网关的浏览器聊天界面

### 多通道消息 (22+ 通道)

- **核心通道**: WhatsApp、Telegram、Slack、Discord、Google Chat、Signal、iMessage、BlueBubbles、IRC、WebChat
- **扩展通道**: Microsoft Teams、Matrix、飞书 (Feishu)、LINE、Mattermost、Nextcloud Talk、Nostr、Synology Chat、Tlon、Twitch、Zalo、Zalo Personal
- **功能特性**: DM 配对/白名单、群组路由、@提及触发、回复标签、按通道分块

### AI 提供商插件 (80+ 扩展)

- **主流提供商**: Anthropic、OpenAI、Google、DeepSeek、Groq、Mistral、Ollama、Hugging Face、Together、Perplexity、xAI、NVIDIA
- **区域提供商**: 火山引擎 (BytePlus/VolcEngine)、千帆 (Qianfan)、MiniMax、月之暗面 (Moonshot)、小米 (Xiaomi)、百炼 (ModelStudio)、Kimi Coding
- **网关代理**: Cloudflare AI Gateway、Vercel AI Gateway、OpenRouter、Copilot Proxy、GitHub Copilot
- **其他**: Amazon Bedrock、Anthropic Vertex、Venice、Chutes、sglang、vLLM

### 伴侣应用

- **macOS 应用** -- SwiftUI 菜单栏应用，包含网关控制、语音唤醒/按键通话、Talk Mode 覆盖层、WebChat、调试工具、远程网关控制
- **iOS 应用** -- SwiftUI 节点应用，支持 Canvas、语音唤醒、Talk Mode、相机、屏幕录制、Bonjour + 设备配对、Watch 应用
- **Android 应用** -- Kotlin/Compose 节点应用，包含 连接/聊天/语音 标签页、Canvas、相机/屏幕录制、设备命令 (通知/位置/短信/照片/联系人/日历/运动/应用更新)

### 工具与自动化

- **浏览器控制** -- 托管的 Chrome/Chromium，支持 CDP、快照、操作、上传、配置文件
- **实时画布 (Canvas)** -- 智能体驱动的可视化工作区，支持 A2UI (Angular/Lit 渲染器)
- **节点 (Nodes)** -- 相机拍照/录像、屏幕录制、位置获取、通知、system.run/system.notify (macOS)
- **定时任务和唤醒** -- 计划智能体任务和 Webhook 触发操作
- **技能平台 (Skills)** -- 内置、托管和工作区技能 (ClawHub 注册表)
- **媒体管道** -- 图像、音频、视频处理；转录钩子；大小限制；临时文件生命周期管理
- **语音** -- 语音唤醒 (macOS/iOS 唤醒词)、Talk Mode (Android 连续语音)、TTS (ElevenLabs + 系统备选)
- **图像生成** -- 通过 fal 等提供商插件
- **网络搜索** -- Brave、DuckDuckGo、Exa、Tavily、Firecrawl、Perplexity
- **记忆系统** -- memory-core + memory-lancedb 扩展，实现持久化智能体记忆
- **智能体间通信** -- sessions_list、sessions_history、sessions_send 实现跨会话协作

### 安全

- **DM 配对** -- 未知发送者收到配对码；基于白名单的访问控制
- **沙箱模式** -- 非主会话 (群组/通道) 使用每会话 Docker 沙箱
- **认证模式** -- Token/密码认证、Tailscale 身份头、浏览器安全加固、SSRF 防护
- **执行审批** -- 可配置的工具执行命令审批策略

## 系统设计

```
消息通道 (WhatsApp / Telegram / Slack / Discord / Google Chat / Signal / iMessage / ...)
               |
               v
+-------------------------------+
|           Gateway             |
|      (WS 控制平面)            |  <-- 配置: ~/.openclaw/openclaw.json
|   ws://127.0.0.1:18789       |  <-- 凭证: ~/.openclaw/credentials/
|                               |  <-- 会话: ~/.openclaw/agents/<id>/sessions/
|  +----------+  +-----------+  |
|  |  通道层  |  |  路由层   |  |
|  +----------+  +-----------+  |
|  +----------+  +-----------+  |
|  |  会话层  |  |  插件层   |  |
|  +----------+  +-----------+  |
|  +----------+  +-----------+  |
|  |  工具层  |  |  定时任务 |  |
|  +----------+  +-----------+  |
|  +----------+  +-----------+  |
|  | Webhooks |  | 控制面板  |  |
|  +----------+  +-----------+  |
+---------------+---------------+
                |
    +-----------+-----------+
    |           |           |
    v           v           v
Pi 智能体     CLI 客户端   伴侣应用
 (RPC)    (openclaw ...)  (macOS/iOS/Android)
```

### 目录结构

```
openclaw/
  src/                  # 核心源代码
    cli/                #   CLI 命令行入口和命令
    commands/           #   命令实现 (308+ 文件)
    gateway/            #   网关服务器、WS 控制平面、认证、会话
    channels/           #   通道路由、契约、共享通道逻辑
    routing/            #   消息路由层
    plugins/            #   插件系统、契约、运行时
    plugin-sdk/         #   插件 SDK (扩展的公共 API 接口)
    agents/             #   智能体运行时、工作区、技能
    media/              #   媒体管道 (图像、音频、视频)
    media-understanding/#   媒体分析 (视觉、转录)
    web-search/         #   网络搜索提供商抽象层
    browser/            #   浏览器自动化 (Playwright/CDP)
    canvas-host/        #   Canvas + A2UI 宿主
    infra/              #   基础设施工具
    security/           #   安全、SSRF、认证原语
    config/             #   配置模式和管理
    hooks/              #   钩子系统
    interactive/        #   交互式提示 UI
    tts/                #   文本转语音
    cron/               #   定时任务管理
    terminal/           #   终端 UI (表格、调色板、进度条)
    tui/                #   终端 UI 模式
    process/            #   进程管理
    daemon/             #   守护进程 (launchd/systemd) 管理
    sessions/           #   会话持久化
    i18n/               #   国际化
    types/              #   共享类型定义
    utils/              #   共享工具函数
  extensions/           # 插件包 (80+ 扩展)
    anthropic/          #   Anthropic 提供商
    openai/             #   OpenAI 提供商
    telegram/           #   Telegram 通道
    discord/            #   Discord 通道
    slack/              #   Slack 通道
    whatsapp/           #   WhatsApp 通道
    ...                 #   (70+ 更多)
  ui/                   # 控制面板 UI (Lit + Vite)
  apps/
    macos/              # macOS 菜单栏应用 (SwiftUI)
    ios/                # iOS 节点应用 (SwiftUI + XcodeGen)
    android/            # Android 节点应用 (Kotlin + Compose)
    shared/             # 跨平台共享代码 (OpenClawKit)
  docs/                 # 文档 (Mintlify, 托管于 docs.openclaw.ai)
  packages/
    clawdbot/           # ClawdBot 包
    moltbot/            # MoltBot 包
  vendor/
    a2ui/               # A2UI 规范 + 渲染器 (Angular, Lit)
  scripts/              # 构建、发布、CI 和工具脚本
  dist/                 # 构建输出
  skills/               # 内置技能
```

### 关键架构决策

- **插件架构**: 所有 AI 提供商和大部分通道都作为 `extensions/` 下的插件实现，通过 Plugin SDK (`openclaw/plugin-sdk/*`) 加载
- **WebSocket 控制平面**: 网关是唯一的协调点；所有客户端 (CLI、应用、UI) 通过 WebSocket 连接
- **本地优先**: 网关运行在用户设备上，绑定到本地回环地址；通过 Tailscale Serve/Funnel 或 SSH 隧道进行远程访问
- **纯 ESM**: 整个代码库使用 ESM TypeScript；动态导入使用专用的 `*.runtime.ts` 边界实现延迟加载
- **Monorepo**: 使用 pnpm workspaces，核心包在根目录，扩展作为工作区包，伴侣应用在 `apps/`

## 本地构建和运行

### 前提条件

- **Node.js** 22.16+ (推荐 Node 24)
- **pnpm** 10.32+ (在 `packageManager` 字段中定义)
- **Bun** (可选，用于直接运行 TypeScript)

### 安装依赖

```bash
git clone https://github.com/openclaw/openclaw.git
cd openclaw
pnpm install
```

### 构建

```bash
pnpm ui:build    # 构建控制面板 UI (首次运行时自动安装 UI 依赖)
pnpm build       # 完整 TypeScript 构建 -> dist/
```

### 开发运行

```bash
# 直接运行 CLI 命令 (通过 tsx 运行 TypeScript，无需构建)
pnpm openclaw ...

# 运行网关
pnpm openclaw gateway --port 18789 --verbose

# 开发循环 (源代码/配置变更时自动重载)
pnpm gateway:watch

# 运行引导向导
pnpm openclaw onboard --install-daemon

# 运行控制面板 UI 开发服务器
pnpm ui:dev
```

### 类型检查、代码检查、格式化

```bash
pnpm tsgo        # TypeScript 类型检查
pnpm check       # 完整检查 (格式化 + 类型检查 + lint + SDK 检查)
pnpm lint        # Oxlint (类型感知)
pnpm format      # Oxfmt (检查模式)
pnpm format:fix  # Oxfmt (自动修复)
```

### 运行测试

```bash
pnpm test                    # 所有单元测试 (Vitest, 并行)
pnpm test -- <path-or-filter> # 指定范围测试
pnpm test:coverage           # 带 V8 覆盖率 (70% 阈值)
pnpm test:e2e                # 端到端测试
pnpm test:live               # 实时测试 (需要真实 API 密钥: OPENCLAW_LIVE_TEST=1)
pnpm test:gateway            # 网关专用测试
pnpm test:ui                 # 控制面板 UI 测试
```

低内存模式:
```bash
OPENCLAW_TEST_PROFILE=low OPENCLAW_TEST_SERIAL_GATEWAY=1 pnpm test
```

### 伴侣应用

```bash
# macOS 应用
pnpm mac:package             # 打包 (当前架构)

# iOS
pnpm ios:gen                 # 生成 Xcode 项目
pnpm ios:build               # 构建模拟器版本
pnpm ios:open                # 在 Xcode 中打开

# Android
pnpm android:install         # 构建并安装到设备
pnpm android:run             # 构建、安装并启动
pnpm android:test            # 运行单元测试
```

### Docker

```bash
# 基于 Docker 的测试
pnpm test:docker:all

# Docker 构建 (优化版，跳过 DTS 生成)
pnpm build:docker
```

### 配置

配置文件: `~/.openclaw/openclaw.json`

```json5
{
  agent: {
    model: "anthropic/claude-opus-4-6",
  },
}
```

完整参考文档: https://docs.openclaw.ai/gateway/configuration
