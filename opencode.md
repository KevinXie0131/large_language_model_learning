# OpenCode - Project Documentation

> AI-powered open-source coding agent. Provider-agnostic alternative to Claude Code.
> Repository: https://github.com/anomalyco/opencode | License: MIT | Version: 1.3.0

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Features & Functionality](#features--functionality)
- [System Design](#system-design)
- [Build & Run Locally](#build--run-locally)
- [中文版本](#中文版本)

---

## Tech Stack

### Languages & Runtime

| Technology | Version | Role |
|---|---|---|
| TypeScript | 5.8.2 | Primary language |
| Bun | 1.3.11 | Package manager & runtime |
| Node.js | 22+ | Server environments |
| Rust | 2024 edition | Tauri desktop app |

### Frontend & UI

| Library | Version | Role |
|---|---|---|
| SolidJS | 1.9.10 | Reactive UI framework (TUI, web, desktop) |
| Vite | 7.1.4 | Build tool & dev server |
| TailwindCSS | 4.1.11 | Utility-first CSS |
| Kobalte | 0.13.11 | Accessible UI components |
| SolidStart | PR build (dfb2020) | Full-stack framework (console) |
| Shiki | 3.20.0 | Syntax highlighting |
| Marked | 17.0.1 | Markdown parsing |
| Virtua | 0.42.3 | Virtual scrolling |
| TanStack Solid Query | 5.91.4 | Async state management |

### Backend & Server

| Library | Version | Role |
|---|---|---|
| Hono | 4.10.7 | HTTP framework (server + Cloudflare Workers) |
| Effect | 4.0.0-beta.35 | Functional effect system |
| Drizzle ORM | 1.0.0-beta.19 | SQL ORM (SQLite local, PlanetScale cloud) |
| Zod | 4.1.8 | Schema validation |
| hono-openapi | 1.1.2 | OpenAPI spec generation |

### AI / LLM Integration

| Library | Version | Role |
|---|---|---|
| Vercel AI SDK (`ai`) | 5.0.124 | Unified multi-provider AI interface |
| @ai-sdk/anthropic | 2.0.65 | Claude provider |
| @ai-sdk/openai | 2.0.89 | OpenAI provider |
| @ai-sdk/google | 2.0.54 | Google Gemini provider |
| @ai-sdk/google-vertex | 3.0.106 | Google Vertex AI provider |
| @ai-sdk/azure | 2.0.91 | Azure OpenAI provider |
| @ai-sdk/amazon-bedrock | 3.0.82 | AWS Bedrock provider |
| @ai-sdk/groq | 2.0.34 | Groq provider |
| @ai-sdk/mistral | 2.0.27 | Mistral provider |
| @ai-sdk/cohere | 2.0.22 | Cohere provider |
| @ai-sdk/xai | 2.0.51 | xAI (Grok) provider |
| @ai-sdk/deepinfra | 1.0.36 | DeepInfra provider |
| @ai-sdk/cerebras | 1.0.36 | Cerebras provider |
| @ai-sdk/togetherai | 1.0.34 | Together AI provider |
| @ai-sdk/perplexity | 2.0.23 | Perplexity provider |
| @ai-sdk/openai-compatible | 1.0.32 | Generic OpenAI-compatible provider |
| @openrouter/ai-sdk-provider | 1.5.4 | OpenRouter provider |
| gitlab-ai-provider | 5.3.1 | GitLab AI provider |
| @modelcontextprotocol/sdk | 1.25.2 | MCP (Model Context Protocol) |

### Desktop / Native

| Library | Version | Role |
|---|---|---|
| Tauri | 2.x | Primary desktop app framework |
| Electron | 40.4.1 | Alternative desktop app framework |
| OpenTUI | 0.1.90 | Terminal UI rendering (SolidJS-based) |

### Infrastructure & DevOps

| Tool | Version | Role |
|---|---|---|
| SST | 3.18.10 | Infrastructure as Code (Cloudflare + AWS) |
| Turbo | 2.8.13 | Monorepo build orchestration |
| Husky | 9.1.7 | Git hooks |
| Playwright | 1.51.0 | E2E testing |
| Prettier | 3.6.2 | Code formatting |
| Cloudflare Workers | - | Serverless compute |
| PlanetScale | 0.4.1 | Cloud MySQL database |
| Stripe | - | Payment processing |

### Utilities

| Library | Version | Role |
|---|---|---|
| Remeda | 2.26.0 | Functional utilities |
| Fuzzysort | 3.1.0 | Fuzzy search |
| ULID | 3.0.1 | Unique ID generation |
| Luxon | 3.6.1 | Date/time handling |
| Clipboardy | 4.0.0 | Clipboard access |
| diff | 8.0.2 | Text diffing |
| Turndown | 7.2.0 | HTML to Markdown |
| Tree-sitter | 0.25.10 | Code parsing (AST) |
| Chokidar | 4.0.3 | File watching |
| @octokit/rest | 22.0.0 | GitHub REST API |
| @octokit/graphql | 9.0.2 | GitHub GraphQL API |

---

## Features & Functionality

### Core Agent System

- **Multi-agent architecture** with distinct modes:
  - `build` - Default agent with full tool access (read, write, execute)
  - `plan` - Read-only analysis mode, disallows edits
  - `general` - Subagent for complex multi-step tasks
  - `explore` - Fast codebase exploration subagent (grep, glob, read)
  - `compaction` - Context window management (hidden)
  - `title` / `summary` - Session metadata generation (hidden)
- **Provider-agnostic**: 20+ LLM providers via Vercel AI SDK
- **Fine-grained permission system**: per-tool, per-directory, per-file-pattern rules
- **Streaming responses**: real-time token streaming with abort support
- **MCP support**: extend capabilities via Model Context Protocol servers
- **Plugin system**: custom skills and tool definitions

### Development Tools

- Bash command execution with streaming output and PTY support
- File read / write / edit with diff-based patching
- Directory navigation and glob-based file search
- Code search with regex support
- Tree-sitter AST parsing for intelligent code navigation
- LSP (Language Server Protocol) integration
- Git integration via Octokit (GitHub REST + GraphQL)
- Web search and web page fetching
- Snapshot and worktree management

### Multi-Platform Interfaces

- **CLI / TUI**: Terminal user interface built with SolidJS + OpenTUI
- **Web UI**: Browser-based interface (SolidJS + Vite)
- **Desktop (Tauri)**: Native macOS/Windows/Linux app with system integration
- **Desktop (Electron)**: Alternative native app
- **API Server**: Hono-based HTTP + WebSocket server

### SaaS Console (Optional)

- Multi-user workspace management
- Authentication via OpenAuth + GitHub
- Session persistence and sync (PlanetScale)
- Real-time collaboration via Cloudflare Durable Objects
- Subscription management via Stripe
- Enterprise deployment support

---

## System Design

### Monorepo Structure

```
opencode/
├── packages/
│   ├── opencode/        # Core: CLI, TUI, server, agent, providers, tools
│   ├── app/             # Web UI (SolidJS + Vite)
│   ├── desktop/         # Tauri desktop app
│   ├── desktop-electron/# Electron desktop app
│   ├── ui/              # Shared UI component library
│   ├── sdk/js/          # Public JavaScript/TypeScript SDK
│   ├── plugin/          # Plugin system (@opencode-ai/plugin)
│   ├── util/            # Shared utilities & error types
│   ├── script/          # Build & utility scripts
│   ├── console/
│   │   ├── app/         # SaaS console frontend (SolidStart)
│   │   ├── core/        # Console backend (Hono + Drizzle)
│   │   ├── mail/        # Email templates (JSX Email)
│   │   ├── resource/    # Multi-environment resource abstraction
│   │   └── function/    # Serverless functions
│   ├── function/        # GitHub webhooks & API (Hono on CF Workers)
│   ├── web/             # Documentation site (Astro)
│   ├── enterprise/      # Enterprise deployment (SolidStart)
│   ├── storybook/       # Component documentation
│   ├── extensions/      # Editor extensions (Zed)
│   ├── containers/      # Docker/container configs
│   ├── identity/        # Identity/auth service
│   ├── slack/           # Slack integration
│   └── docs/            # Generated documentation
├── infra/               # SST infrastructure definitions
├── specs/               # API specifications & schemas
├── nix/                 # NixOS packaging
├── sdks/                # SDK distributions
└── .github/             # CI/CD workflows
```

### Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    User Interfaces                        │
│  ┌─────────┐   ┌───────────┐   ┌─────────────────────┐  │
│  │   TUI   │   │  Web UI   │   │  Desktop (Tauri/E)  │  │
│  │ OpenTUI │   │ SolidJS   │   │    SolidJS + Native │  │
│  │ SolidJS │   │ + Vite    │   │    APIs             │  │
│  └────┬────┘   └─────┬─────┘   └──────────┬──────────┘  │
│       └──────────────┬┘────────────────────┘             │
└──────────────────────┼───────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────┼───────────────────────────────────┐
│              Hono Server (packages/opencode)              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Routes: /session  /project  /file  /config  /mcp   │ │
│  │          /provider /tui  /pty  /event  /permission  │ │
│  └───────────────────────┬─────────────────────────────┘ │
│  ┌───────────┐ ┌────────┴────────┐ ┌─────────────────┐  │
│  │   Agent   │ │   Tool System   │ │    Permission   │  │
│  │  System   │ │  bash, read,    │ │    Framework    │  │
│  │ build/plan│ │  write, edit,   │ │  per-tool/dir/  │  │
│  │ general/  │ │  glob, grep,    │ │  file-pattern   │  │
│  │ explore   │ │  web, mcp, ...  │ │                 │  │
│  └─────┬─────┘ └─────────────────┘ └─────────────────┘  │
│        │                                                  │
│  ┌─────┴─────────────────────────────────────────────┐   │
│  │         Provider Abstraction (Vercel AI SDK)       │   │
│  │  Claude | OpenAI | Gemini | Azure | Bedrock | ...  │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────────┐
│                 Data Layer                                │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │  SQLite  │  │  PlanetScale  │  │  Cloudflare R2   │  │
│  │ (local)  │  │  (cloud MySQL)│  │  (file storage)  │  │
│  │  Drizzle │  │   Drizzle     │  │                  │  │
│  └──────────┘  └───────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **SolidJS everywhere** - Same reactive framework across TUI, web, and desktop for code reuse
2. **Provider-agnostic AI** - Vercel AI SDK provides a single interface for 20+ LLM providers
3. **Local-first** - CLI with local SQLite; cloud features are optional
4. **Hono for HTTP** - Lightweight, runs on both Bun and Cloudflare Workers
5. **Effect for side effects** - Functional effect system for complex async workflows
6. **Drizzle for data** - Type-safe ORM works with SQLite (local) and MySQL (cloud)
7. **SST for infra** - Declarative infrastructure targeting Cloudflare as primary cloud
8. **Turbo for builds** - Incremental builds with caching across the monorepo
9. **Fine-grained permissions** - Agent tool access is controlled per-tool, per-directory, per-file-pattern
10. **Conditional imports** - `#db` import map switches between Bun SQLite and Node.js drivers

### Server Routes

| Route | Purpose |
|---|---|
| `/session` | Session CRUD, message streaming, agent execution |
| `/project` | Project metadata, git operations |
| `/file` | File read/write/watch |
| `/config` | Runtime configuration |
| `/mcp` | MCP server management |
| `/provider` | AI provider & model management |
| `/tui` | TUI-specific operations |
| `/pty` | Terminal emulation (PTY) |
| `/event` | SSE event stream |
| `/permission` | Permission rule management |
| `/question` | User interaction prompts |
| `/global` | Global state & settings |

### Database Schema

- Uses Drizzle ORM with snake_case column naming convention
- Local storage: Bun-native SQLite (`~/.local/share/opencode/opencode.db`)
- Cloud storage: PlanetScale (MySQL) for console/enterprise
- Migrations managed via `drizzle-kit`

---

## Build & Run Locally

### Prerequisites

- **Bun** >= 1.3.11 - https://bun.sh/docs/installation
- **Node.js** >= 22 (for some tooling)
- **Rust** (only if building the Tauri desktop app)
- **Git**

### Install Dependencies

```bash
git clone https://github.com/anomalyco/opencode
cd opencode
bun install
```

### Development Commands

```bash
# CLI / TUI (interactive terminal mode)
bun run dev              # Run against packages/opencode directory
bun run dev <dir>        # Run against a specific directory
bun run dev .            # Run against repo root

# Web UI
bun run dev:web          # Start web interface + API server

# Desktop App (Tauri)
bun run dev:desktop      # Start native desktop app (requires Rust)

# SaaS Console
bun run dev:console      # Start console frontend + backend

# Storybook
bun run dev:storybook    # Component documentation
```

### Type Checking

```bash
# All packages
bun turbo typecheck

# Single package (always run from package directory, never use tsc directly)
cd packages/opencode && bun run typecheck
cd packages/app && bun run typecheck
```

### Testing

```bash
# Tests CANNOT be run from repo root (guarded)
# Always run from the specific package directory

cd packages/opencode && bun test
cd packages/app && bun run test:unit
cd packages/app && bun run test:e2e        # Playwright E2E tests
```

### Building

```bash
# Build all packages
bun turbo build

# Build CLI binary (single platform)
cd packages/opencode && bun run build
# Output: dist/opencode-<platform>/bin/opencode
# Platforms: darwin-arm64, darwin-x64, linux-x64, linux-arm64, windows-x64
```

### Database Operations

```bash
cd packages/opencode
bun run db <command>        # Drizzle-kit commands (generate, migrate, push, etc.)
```

### SDK Regeneration

```bash
./packages/sdk/js/script/build.ts
```

### Environment

- Local config: `~/.opencode/config.json`
- Database: `~/.local/share/opencode/opencode.db` (SQLite)
- Custom DB path: set `OPENCODE_DB` environment variable
- Default branch: `dev` (not `main`)

### Code Style

- `semi: false`, `printWidth: 120` (Prettier)
- Single-word variable names preferred (`cfg`, `dir`, `err`, `opts`)
- Avoid `try`/`catch`, prefer Result types
- No `any` types
- Prefer Bun APIs (`Bun.file()`)
- Functional array methods over loops
- `const` over `let`, ternaries over reassignment
- Early returns, avoid `else`
- snake_case for Drizzle schema fields

---

---

# 中文版本

# OpenCode - 项目文档

> AI 驱动的开源编码代理。与 Claude Code 功能类似的跨供应商替代方案。
> 仓库地址：https://github.com/anomalyco/opencode | 许可证：MIT | 版本：1.3.0

---

## 目录

- [技术栈](#技术栈)
- [功能与特性](#功能与特性)
- [系统设计](#系统设计)
- [本地构建与运行](#本地构建与运行)

---

## 技术栈

### 语言与运行时

| 技术 | 版本 | 用途 |
|---|---|---|
| TypeScript | 5.8.2 | 主要开发语言 |
| Bun | 1.3.11 | 包管理器与运行时 |
| Node.js | 22+ | 服务端环境 |
| Rust | 2024 版 | Tauri 桌面应用 |

### 前端与 UI

| 库 | 版本 | 用途 |
|---|---|---|
| SolidJS | 1.9.10 | 响应式 UI 框架（TUI、Web、桌面端通用） |
| Vite | 7.1.4 | 构建工具与开发服务器 |
| TailwindCSS | 4.1.11 | 原子化 CSS 框架 |
| Kobalte | 0.13.11 | 无障碍 UI 组件库 |
| SolidStart | PR 构建版 (dfb2020) | 全栈框架（控制台） |
| Shiki | 3.20.0 | 语法高亮 |
| Marked | 17.0.1 | Markdown 解析 |
| Virtua | 0.42.3 | 虚拟滚动 |
| TanStack Solid Query | 5.91.4 | 异步状态管理 |

### 后端与服务器

| 库 | 版本 | 用途 |
|---|---|---|
| Hono | 4.10.7 | HTTP 框架（服务器 + Cloudflare Workers） |
| Effect | 4.0.0-beta.35 | 函数式副作用系统 |
| Drizzle ORM | 1.0.0-beta.19 | SQL ORM（本地 SQLite，云端 PlanetScale） |
| Zod | 4.1.8 | 数据模式验证 |
| hono-openapi | 1.1.2 | OpenAPI 规范生成 |

### AI / LLM 集成

| 库 | 版本 | 用途 |
|---|---|---|
| Vercel AI SDK (`ai`) | 5.0.124 | 统一多供应商 AI 接口 |
| @ai-sdk/anthropic | 2.0.65 | Claude 供应商 |
| @ai-sdk/openai | 2.0.89 | OpenAI 供应商 |
| @ai-sdk/google | 2.0.54 | Google Gemini 供应商 |
| @ai-sdk/google-vertex | 3.0.106 | Google Vertex AI 供应商 |
| @ai-sdk/azure | 2.0.91 | Azure OpenAI 供应商 |
| @ai-sdk/amazon-bedrock | 3.0.82 | AWS Bedrock 供应商 |
| @ai-sdk/groq | 2.0.34 | Groq 供应商 |
| @ai-sdk/mistral | 2.0.27 | Mistral 供应商 |
| @ai-sdk/cohere | 2.0.22 | Cohere 供应商 |
| @ai-sdk/xai | 2.0.51 | xAI (Grok) 供应商 |
| @ai-sdk/deepinfra | 1.0.36 | DeepInfra 供应商 |
| @ai-sdk/cerebras | 1.0.36 | Cerebras 供应商 |
| @ai-sdk/togetherai | 1.0.34 | Together AI 供应商 |
| @ai-sdk/perplexity | 2.0.23 | Perplexity 供应商 |
| @ai-sdk/openai-compatible | 1.0.32 | 通用 OpenAI 兼容供应商 |
| @openrouter/ai-sdk-provider | 1.5.4 | OpenRouter 供应商 |
| gitlab-ai-provider | 5.3.1 | GitLab AI 供应商 |
| @modelcontextprotocol/sdk | 1.25.2 | MCP（模型上下文协议） |

### 桌面端 / 原生应用

| 库 | 版本 | 用途 |
|---|---|---|
| Tauri | 2.x | 主要桌面应用框架 |
| Electron | 40.4.1 | 备选桌面应用框架 |
| OpenTUI | 0.1.90 | 终端 UI 渲染（基于 SolidJS） |

### 基础设施与 DevOps

| 工具 | 版本 | 用途 |
|---|---|---|
| SST | 3.18.10 | 基础设施即代码（Cloudflare + AWS） |
| Turbo | 2.8.13 | Monorepo 构建编排 |
| Husky | 9.1.7 | Git Hooks |
| Playwright | 1.51.0 | 端到端测试 |
| Prettier | 3.6.2 | 代码格式化 |
| Cloudflare Workers | - | 无服务器计算 |
| PlanetScale | 0.4.1 | 云端 MySQL 数据库 |
| Stripe | - | 支付处理 |

### 工具库

| 库 | 版本 | 用途 |
|---|---|---|
| Remeda | 2.26.0 | 函数式工具库 |
| Fuzzysort | 3.1.0 | 模糊搜索 |
| ULID | 3.0.1 | 唯一 ID 生成 |
| Luxon | 3.6.1 | 日期时间处理 |
| Clipboardy | 4.0.0 | 剪贴板访问 |
| diff | 8.0.2 | 文本差异比较 |
| Turndown | 7.2.0 | HTML 转 Markdown |
| Tree-sitter | 0.25.10 | 代码解析（AST） |
| Chokidar | 4.0.3 | 文件监听 |
| @octokit/rest | 22.0.0 | GitHub REST API |
| @octokit/graphql | 9.0.2 | GitHub GraphQL API |

---

## 功能与特性

### 核心代理系统

- **多代理架构**，具备不同模式：
  - `build` - 默认代理，拥有完整工具访问权限（读取、写入、执行）
  - `plan` - 只读分析模式，禁止编辑操作
  - `general` - 子代理，用于复杂的多步骤任务
  - `explore` - 快速代码库探索子代理（grep、glob、read）
  - `compaction` - 上下文窗口管理（内部使用）
  - `title` / `summary` - 会话元数据生成（内部使用）
- **供应商无关**：通过 Vercel AI SDK 支持 20+ LLM 供应商
- **细粒度权限系统**：按工具、按目录、按文件模式的权限规则
- **流式响应**：实时 token 流，支持中止操作
- **MCP 支持**：通过模型上下文协议服务器扩展能力
- **插件系统**：自定义技能和工具定义

### 开发工具

- Bash 命令执行，支持流式输出和 PTY
- 文件读取/写入/编辑，基于 diff 的补丁更新
- 目录导航和 glob 模式文件搜索
- 正则表达式代码搜索
- Tree-sitter AST 解析，智能代码导航
- LSP（语言服务器协议）集成
- 通过 Octokit 集成 Git（GitHub REST + GraphQL）
- 网页搜索和网页内容抓取
- 快照和工作树管理

### 多平台界面

- **CLI / TUI**：基于 SolidJS + OpenTUI 的终端用户界面
- **Web UI**：浏览器界面（SolidJS + Vite）
- **桌面端（Tauri）**：原生 macOS/Windows/Linux 应用，集成系统功能
- **桌面端（Electron）**：备选原生应用方案
- **API 服务器**：基于 Hono 的 HTTP + WebSocket 服务器

### SaaS 控制台（可选）

- 多用户工作区管理
- 通过 OpenAuth + GitHub 认证
- 会话持久化与同步（PlanetScale）
- 通过 Cloudflare Durable Objects 实时协作
- 通过 Stripe 管理订阅
- 企业级部署支持

---

## 系统设计

### Monorepo 结构

```
opencode/
├── packages/
│   ├── opencode/        # 核心：CLI、TUI、服务器、代理、供应商、工具
│   ├── app/             # Web UI（SolidJS + Vite）
│   ├── desktop/         # Tauri 桌面应用
│   ├── desktop-electron/# Electron 桌面应用
│   ├── ui/              # 共享 UI 组件库
│   ├── sdk/js/          # 公开 JavaScript/TypeScript SDK
│   ├── plugin/          # 插件系统（@opencode-ai/plugin）
│   ├── util/            # 共享工具库与错误类型
│   ├── script/          # 构建与工具脚本
│   ├── console/
│   │   ├── app/         # SaaS 控制台前端（SolidStart）
│   │   ├── core/        # 控制台后端（Hono + Drizzle）
│   │   ├── mail/        # 邮件模板（JSX Email）
│   │   ├── resource/    # 多环境资源抽象
│   │   └── function/    # 无服务器函数
│   ├── function/        # GitHub Webhooks 与 API（Hono on CF Workers）
│   ├── web/             # 文档网站（Astro）
│   ├── enterprise/      # 企业部署版（SolidStart）
│   ├── storybook/       # 组件文档
│   ├── extensions/      # 编辑器扩展（Zed）
│   ├── containers/      # Docker/容器配置
│   ├── identity/        # 身份认证服务
│   ├── slack/           # Slack 集成
│   └── docs/            # 生成的文档
├── infra/               # SST 基础设施定义
├── specs/               # API 规范与 Schema
├── nix/                 # NixOS 打包
├── sdks/                # SDK 分发
└── .github/             # CI/CD 工作流
```

### 架构概览

```
┌──────────────────────────────────────────────────────────┐
│                       用户界面                            │
│  ┌─────────┐   ┌───────────┐   ┌─────────────────────┐  │
│  │   TUI   │   │  Web UI   │   │ 桌面端 (Tauri/E)    │  │
│  │ OpenTUI │   │ SolidJS   │   │ SolidJS + 原生 API  │  │
│  │ SolidJS │   │ + Vite    │   │                     │  │
│  └────┬────┘   └─────┬─────┘   └──────────┬──────────┘  │
│       └──────────────┬┘────────────────────┘             │
└──────────────────────┼───────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────┼───────────────────────────────────┐
│              Hono 服务器 (packages/opencode)               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  路由: /session  /project  /file  /config  /mcp     │ │
│  │        /provider /tui  /pty  /event  /permission    │ │
│  └───────────────────────┬─────────────────────────────┘ │
│  ┌───────────┐ ┌────────┴────────┐ ┌─────────────────┐  │
│  │  代理系统  │ │    工具系统      │ │    权限框架     │  │
│  │  build/   │ │  bash, read,    │ │  按工具/目录/   │  │
│  │  plan/    │ │  write, edit,   │ │  文件模式控制   │  │
│  │  general/ │ │  glob, grep,    │ │                 │  │
│  │  explore  │ │  web, mcp, ...  │ │                 │  │
│  └─────┬─────┘ └─────────────────┘ └─────────────────┘  │
│        │                                                  │
│  ┌─────┴─────────────────────────────────────────────┐   │
│  │       供应商抽象层 (Vercel AI SDK)                  │   │
│  │  Claude | OpenAI | Gemini | Azure | Bedrock | ... │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────────┐
│                    数据层                                  │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │  SQLite  │  │  PlanetScale  │  │  Cloudflare R2   │  │
│  │（本地）   │  │（云端 MySQL）  │  │（文件存储）       │  │
│  │  Drizzle │  │   Drizzle     │  │                  │  │
│  └──────────┘  └───────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 关键设计决策

1. **SolidJS 全平台复用** - TUI、Web、桌面端使用相同的响应式框架，最大化代码复用
2. **供应商无关的 AI 接入** - Vercel AI SDK 提供统一接口，支持 20+ LLM 供应商
3. **本地优先** - CLI 使用本地 SQLite，云端功能可选
4. **Hono 作为 HTTP 框架** - 轻量级，同时支持 Bun 和 Cloudflare Workers
5. **Effect 处理副作用** - 函数式副作用系统管理复杂异步流程
6. **Drizzle 数据访问** - 类型安全 ORM，支持 SQLite（本地）和 MySQL（云端）
7. **SST 基础设施** - 声明式基础设施，以 Cloudflare 为主要云平台
8. **Turbo 构建编排** - 增量构建与缓存，覆盖整个 monorepo
9. **细粒度权限** - 代理工具访问按工具、目录、文件模式精确控制
10. **条件导入** - `#db` 导入映射在 Bun SQLite 和 Node.js 驱动间自动切换

### 服务器路由

| 路由 | 用途 |
|---|---|
| `/session` | 会话 CRUD、消息流、代理执行 |
| `/project` | 项目元数据、Git 操作 |
| `/file` | 文件读写与监听 |
| `/config` | 运行时配置 |
| `/mcp` | MCP 服务器管理 |
| `/provider` | AI 供应商与模型管理 |
| `/tui` | TUI 专用操作 |
| `/pty` | 终端模拟（PTY） |
| `/event` | SSE 事件流 |
| `/permission` | 权限规则管理 |
| `/question` | 用户交互提示 |
| `/global` | 全局状态与设置 |

### 数据库模式

- 使用 Drizzle ORM，列名采用 snake_case 命名规范
- 本地存储：Bun 原生 SQLite（`~/.local/share/opencode/opencode.db`）
- 云端存储：PlanetScale（MySQL），用于控制台/企业版
- 迁移通过 `drizzle-kit` 管理

---

## 本地构建与运行

### 前置条件

- **Bun** >= 1.3.11 - https://bun.sh/docs/installation
- **Node.js** >= 22（部分工具需要）
- **Rust**（仅构建 Tauri 桌面应用时需要）
- **Git**

### 安装依赖

```bash
git clone https://github.com/anomalyco/opencode
cd opencode
bun install
```

### 开发命令

```bash
# CLI / TUI（交互式终端模式）
bun run dev              # 在 packages/opencode 目录下运行
bun run dev <dir>        # 指定目录运行
bun run dev .            # 在仓库根目录运行

# Web UI
bun run dev:web          # 启动 Web 界面 + API 服务器

# 桌面应用（Tauri）
bun run dev:desktop      # 启动原生桌面应用（需要 Rust）

# SaaS 控制台
bun run dev:console      # 启动控制台前后端

# Storybook
bun run dev:storybook    # 组件文档
```

### 类型检查

```bash
# 所有包
bun turbo typecheck

# 单个包（始终在包目录内运行，不要直接使用 tsc）
cd packages/opencode && bun run typecheck
cd packages/app && bun run typecheck
```

### 测试

```bash
# 测试不能从仓库根目录运行（有保护机制）
# 始终在对应的包目录内运行

cd packages/opencode && bun test
cd packages/app && bun run test:unit
cd packages/app && bun run test:e2e        # Playwright 端到端测试
```

### 构建

```bash
# 构建所有包
bun turbo build

# 构建 CLI 二进制文件（当前平台）
cd packages/opencode && bun run build
# 输出：dist/opencode-<platform>/bin/opencode
# 支持平台：darwin-arm64、darwin-x64、linux-x64、linux-arm64、windows-x64
```

### 数据库操作

```bash
cd packages/opencode
bun run db <command>        # Drizzle-kit 命令（generate、migrate、push 等）
```

### SDK 重新生成

```bash
./packages/sdk/js/script/build.ts
```

### 环境配置

- 本地配置：`~/.opencode/config.json`
- 数据库：`~/.local/share/opencode/opencode.db`（SQLite）
- 自定义数据库路径：设置 `OPENCODE_DB` 环境变量
- 默认分支：`dev`（不是 `main`）

### 代码风格

- `semi: false`，`printWidth: 120`（Prettier）
- 优先使用单词变量名（`cfg`、`dir`、`err`、`opts`）
- 避免 `try`/`catch`，优先使用 Result 类型
- 禁止使用 `any` 类型
- 优先使用 Bun API（`Bun.file()`）
- 函数式数组方法优先于循环
- `const` 优先于 `let`，三元表达式优先于重新赋值
- 使用提前返回，避免 `else`
- Drizzle Schema 字段使用 snake_case
