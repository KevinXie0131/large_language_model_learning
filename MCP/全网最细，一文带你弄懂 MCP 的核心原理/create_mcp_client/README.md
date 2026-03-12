# MCP Client（Model Context Protocol 客户端）

一个基于 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 协议的 Node.js 客户端，使用 OpenAI SDK 兼容多种大语言模型，能够连接任意 MCP 服务器并智能调用其提供的工具。

## 什么是 MCP？

MCP（Model Context Protocol，模型上下文协议）是一个开放协议，它标准化了应用程序如何向大语言模型（LLM）提供上下文。你可以把 MCP 想象成 AI 应用的"USB-C 接口"——它提供了一种标准化的方式，将 AI 模型连接到不同的数据源和工具。

### MCP 核心架构

MCP 采用客户端-服务器架构：

```
┌─────────────────────────────────────────┐
│           MCP 主机（本项目）               │
│                                         │
│  ┌─────────────┐  ┌─────────────┐       │
│  │ MCP Client 1│  │ MCP Client 2│  ...  │
│  └──────┬──────┘  └──────┬──────┘       │
└─────────┼────────────────┼──────────────┘
          │                │
    ┌─────┴─────┐    ┌─────┴─────┐
    │MCP Server │    │MCP Server │
    │  (本地)    │    │  (远程)    │
    └───────────┘    └───────────┘
```

- **MCP 主机（Host）**：协调和管理 MCP 客户端的 AI 应用程序
- **MCP 客户端（Client）**：与 MCP 服务器保持连接，获取上下文信息
- **MCP 服务器（Server）**：向客户端提供工具、资源和提示等上下文

### MCP 核心原语（Primitives）

MCP 服务器可以提供三种核心原语：

| 原语 | 说明 | 示例 |
|------|------|------|
| **工具（Tools）** | LLM 可以调用的可执行函数 | 文件操作、API 调用、数据库查询 |
| **资源（Resources）** | 为 LLM 提供上下文的数据源 | 文件内容、数据库记录、API 响应 |
| **提示（Prompts）** | 可复用的交互模板 | 系统提示词、Few-shot 示例 |

## 功能特性

- 支持连接任意 MCP 服务器（Node.js / Python）
- 使用 OpenAI SDK，兼容所有 OpenAI 接口兼容的模型服务
- 自动发现并注册 MCP 服务器提供的工具
- 支持 LLM 自动决策是否调用工具（Function Calling）
- 支持多轮对话，保持上下文连贯
- 交互式命令行聊天界面
- 完善的错误处理和中文提示

## 兼容的模型服务

由于使用 OpenAI SDK，本项目兼容所有提供 OpenAI 兼容接口的服务：

| 服务商 | BASE_URL | 模型示例 |
|--------|----------|----------|
| OpenAI | `https://api.openai.com/v1`（默认） | `gpt-4o`, `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |
| Ollama（本地） | `http://localhost:11434/v1` | `llama3`, `qwen2` |
| Azure OpenAI | 你的 Azure 端点 | 你部署的模型 |
| 其他兼容服务 | 对应的 API 地址 | 对应的模型名称 |

## 快速开始

### 环境要求

- Node.js >= 18
- npm 或 yarn

### 安装步骤

1. **克隆或下载项目**

```bash
cd create_mcp_client
```

2. **安装依赖**

```bash
npm install
```

3. **配置环境变量**

复制 `.env.example` 为 `.env` 并填写你的配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# 必填：API 密钥
OPENAI_API_KEY=your-api-key-here

# 可选：自定义 API 地址（使用其他兼容服务时需要配置）
# OPENAI_BASE_URL=https://api.deepseek.com/v1

# 可选：指定模型名称（默认 gpt-4o）
# MODEL_NAME=deepseek-chat
```

4. **编译 TypeScript**

```bash
npm run build
```

5. **运行客户端**

本客户端支持两种运行模式：

**模式一：配置文件模式（推荐）**

```bash
node build/index.js <服务器名> <配置文件路径>
```

**模式二：直接脚本模式**

```bash
node build/index.js <服务器脚本路径>
```

## 配置文件

推荐使用 JSON 配置文件管理 MCP 服务器，格式兼容 Claude Desktop 的配置风格。

创建 `mcp-servers.json`：

```json
{
  "mcpServers": {
    "time": {
      "command": "node",
      "args": ["C:/path/to/time-server/dist/index.js"],
      "description": "获取当前时间的 MCP 服务器"
    },
    "mongodb": {
      "command": "npx",
      "args": ["mcp-mongo-server", "mongodb://localhost:27017/mydb"],
      "description": "MongoDB 数据库操作"
    },
    "weather": {
      "command": "python",
      "args": ["C:/path/to/weather-server/main.py"],
      "env": {
        "WEATHER_API_KEY": "your-weather-api-key"
      }
    }
  },
  "defaultServer": "time",
  "system": "你是一个有用的 AI 助手，请用中文回复。"
}
```

### 配置字段说明

| 字段 | 说明 |
|------|------|
| `mcpServers` | 服务器配置映射，键为服务器名称 |
| `mcpServers.*.command` | 启动服务器的命令（如 `node`, `npx`, `python`） |
| `mcpServers.*.args` | 传递给命令的参数列表 |
| `mcpServers.*.description` | 服务器描述（可选） |
| `mcpServers.*.env` | 额外的环境变量（可选，会与当前环境变量合并） |
| `defaultServer` | 默认使用的服务器名称（可选） |
| `system` | 系统提示词，用于指导 LLM 的行为（可选） |

## 使用示例

### 通过配置文件连接

```bash
$ node build/index.js time ./mcp-servers.json

正在连接服务器 "time"...
服务器描述: 获取当前时间的 MCP 服务器
已连接到 MCP 服务器，可用工具: [ 'get_current_time' ]

=== MCP 客户端已启动 ===
当前模型: gpt-4o
系统提示词: 你是一个有用的 AI 助手，请用中文回复。
输入你的问题开始对话，输入 "quit" 退出，输入 "clear" 清除历史。

你: 现在几点了？

[调用工具 get_current_time，参数: {}]

助手: 现在是 2025 年 3 月 12 日下午 4:30。
```

### 直接连接脚本

```bash
$ node build/index.js ./weather-server/build/index.js

已连接到 MCP 服务器，可用工具: [ 'get_weather', 'get_forecast' ]

=== MCP 客户端已启动 ===
当前模型: gpt-4o
输入你的问题开始对话，输入 "quit" 退出，输入 "clear" 清除历史。

你: 北京今天天气怎么样？

[调用工具 get_weather，参数: {"location":"北京"}]

助手: 北京今天的天气是晴天，气温 25°C，湿度 45%，微风。适合户外活动！

你: clear

对话历史已清除。

你: quit

再见！
```

### 交互命令

| 命令 | 说明 |
|------|------|
| 任意文本 | 发送查询给 LLM |
| `clear` | 清除对话历史，开始新会话 |
| `quit` / `exit` | 退出程序 |

## 工作原理

当你发送一条查询时，客户端会执行以下流程：

```
用户输入
  │
  ▼
┌─────────────────┐
│  添加到对话历史   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  发送给 LLM      │────▶│  LLM 分析并决策   │
│  (附带工具列表)   │     │  是否需要调用工具  │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
           ┌──────────────┐          ┌──────────────┐
           │ 不需要工具     │          │ 需要调用工具   │
           │ 直接返回回复   │          │              │
           └──────────────┘          └───────┬──────┘
                                             │
                                             ▼
                                    ┌──────────────┐
                                    │ MCP 执行工具   │
                                    │ 返回结果       │
                                    └───────┬──────┘
                                             │
                                             ▼
                                    ┌──────────────┐
                                    │ 结果发送给 LLM │
                                    │ 生成最终回复   │
                                    └──────────────┘
```

### 详细步骤

1. **工具发现**：客户端连接 MCP 服务器后，通过 `tools/list` 获取所有可用工具及其参数定义
2. **格式转换**：将 MCP 工具格式转换为 OpenAI Function Calling 格式
3. **查询处理**：将用户消息和工具列表一起发送给 LLM
4. **工具调用**：如果 LLM 决定使用工具，客户端通过 MCP 协议的 `tools/call` 执行工具
5. **结果整合**：将工具执行结果返回给 LLM，由其生成最终的自然语言回复
6. **上下文保持**：所有消息（用户、助手、工具结果）都保存在对话历史中，支持多轮对话

## 项目结构

```
create_mcp_client/
├── src/
│   └── index.ts              # 主要源代码（MCP 客户端实现）
├── build/                    # 编译输出目录（自动生成）
├── mcp-servers.example.json  # MCP 服务器配置文件示例
├── package.json              # 项目配置和依赖
├── tsconfig.json             # TypeScript 编译配置
├── .env.example              # 环境变量示例
├── .env                      # 环境变量配置（需自行创建）
├── .gitignore                # Git 忽略规则
└── README.md                 # 项目说明文档
```

## 常见问题

### 连接 MCP 服务器失败

- 确认服务器脚本路径正确
- 使用绝对路径如果相对路径不起作用
- Windows 用户请使用正斜杠 `/` 或转义反斜杠 `\\`
- 确认服务器文件扩展名正确（.js 或 .py）
- Python 服务器需要确保 Python 已安装并在 PATH 中

### API 调用失败

- 检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确
- 如果使用第三方服务，确认 `OPENAI_BASE_URL` 配置正确
- 确认模型名称 `MODEL_NAME` 与你使用的服务匹配
- 检查网络连接和 API 配额

### 工具调用失败

- 确认 MCP 服务器正常运行
- 检查工具所需的环境变量是否已设置
- 查看控制台错误信息获取详细原因

### 首次响应较慢

- 首次响应可能需要 10-30 秒，这是正常现象
- 原因：服务器初始化 + LLM 处理 + 工具执行
- 后续响应通常会更快

## 开发指南

### 本地开发

```bash
# 安装依赖
npm install

# 编译并运行
npm run build && node build/index.js <服务器路径>
```

### 扩展功能

如果你想扩展此客户端，以下是一些方向：

- **支持多服务器连接**：同时连接多个 MCP 服务器，合并工具列表
- **支持 Streamable HTTP 传输**：除了 stdio，还支持远程 MCP 服务器
- **添加系统提示词**：在对话开始时注入自定义的系统提示词
- **流式输出**：使用 OpenAI 的流式 API 实现打字机效果
- **对话持久化**：将对话历史保存到文件，支持恢复会话

## 技术栈

- **TypeScript** - 类型安全的 JavaScript 超集
- **OpenAI SDK** - 与兼容 OpenAI 接口的 LLM 服务交互
- **@modelcontextprotocol/sdk** - MCP 协议的官方 TypeScript SDK
- **dotenv** - 环境变量管理

## 许可证

MIT
