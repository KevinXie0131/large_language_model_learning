/**
 * MCP Client - 基于 Model Context Protocol 的智能客户端
 *
 * 本客户端实现了 MCP（Model Context Protocol）协议，能够连接到任意 MCP 服务器，
 * 并通过兼容 OpenAI 接口的大语言模型来智能地调用服务器提供的工具。
 *
 * 核心功能：
 * 1. 连接 MCP 服务器并发现可用工具
 * 2. 将用户查询发送给 LLM，由 LLM 决定是否调用工具
 * 3. 执行工具调用并将结果返回给 LLM 生成最终回复
 * 4. 支持多轮对话和交互式聊天
 *
 * 架构说明：
 * - 使用 @modelcontextprotocol/sdk 处理 MCP 协议通信
 * - 使用 OpenAI SDK（兼容 OpenAI 接口的任意服务）处理 LLM 交互
 * - 通过 stdio 传输层与 MCP 服务器通信
 */

import OpenAI from "openai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import readline from "readline/promises";
import dotenv from "dotenv";

import type {
  ChatCompletionMessageParam,
  ChatCompletionFunctionTool,
} from "openai/resources/chat/completions/completions.js";

// 加载环境变量配置
dotenv.config();

// ============================================================================
// 环境变量校验
// ============================================================================

/** OpenAI API 密钥（必填） */
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  throw new Error(
    "OPENAI_API_KEY 未设置，请在 .env 文件中配置，或设置环境变量"
  );
}

/** API 基础地址（可选，默认使用 OpenAI 官方地址） */
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || undefined;

/** 模型名称（可选，默认使用 gpt-4o） */
const MODEL_NAME = process.env.MODEL_NAME || "gpt-4o";

// ============================================================================
// MCP 客户端类
// ============================================================================

/**
 * MCPClient - MCP 协议客户端
 *
 * 该类封装了 MCP 客户端的核心逻辑，包括：
 * - 与 MCP 服务器建立连接
 * - 发现并注册服务器提供的工具
 * - 通过 LLM 处理用户查询
 * - 执行工具调用并返回结果
 * - 维护多轮对话上下文
 */
class MCPClient {
  /** MCP 客户端实例，用于与 MCP 服务器通信 */
  private mcp: Client;

  /** OpenAI SDK 实例，用于与大语言模型交互 */
  private openai: OpenAI;

  /** stdio 传输层，用于进程间通信 */
  private transport: StdioClientTransport | null = null;

  /** 从 MCP 服务器获取的可用工具列表（OpenAI Function Calling 格式） */
  private tools: ChatCompletionFunctionTool[] = [];

  /** 多轮对话的消息历史记录 */
  private messageHistory: ChatCompletionMessageParam[] = [];

  constructor() {
    // 初始化 OpenAI 客户端
    this.openai = new OpenAI({
      apiKey: OPENAI_API_KEY,
      ...(OPENAI_BASE_URL && { baseURL: OPENAI_BASE_URL }),
    });

    // 初始化 MCP 客户端，设置客户端名称和版本
    this.mcp = new Client({
      name: "mcp-client-cli",
      version: "1.0.0",
    });
  }

  // --------------------------------------------------------------------------
  // 服务器连接管理
  // --------------------------------------------------------------------------

  /**
   * 连接到 MCP 服务器
   *
   * 根据服务器脚本的类型（.js 或 .py）自动选择合适的运行命令，
   * 通过 stdio 传输层建立与 MCP 服务器的连接，并获取可用的工具列表。
   *
   * MCP 连接生命周期：
   * 1. 创建 stdio 传输层
   * 2. 客户端与服务器握手（初始化 + 能力协商）
   * 3. 获取服务器提供的工具列表
   * 4. 将工具转换为 OpenAI 格式以供 LLM 使用
   *
   * @param serverScriptPath - MCP 服务器脚本的路径（支持 .js 和 .py 文件）
   * @throws 当脚本路径不是 .js 或 .py 文件时抛出错误
   */
  async connectToServer(serverScriptPath: string): Promise<void> {
    try {
      // 判断服务器脚本类型
      const isJs = serverScriptPath.endsWith(".js");
      const isPy = serverScriptPath.endsWith(".py");
      if (!isJs && !isPy) {
        throw new Error("服务器脚本必须是 .js 或 .py 文件");
      }

      // 根据脚本类型确定启动命令
      // Python 脚本在 Windows 上使用 python，其他平台使用 python3
      // JavaScript 脚本使用当前 Node.js 运行时
      const command = isPy
        ? process.platform === "win32"
          ? "python"
          : "python3"
        : process.execPath;

      // 创建 stdio 传输层
      // stdio 传输是 MCP 的本地通信方式，通过标准输入/输出流进行进程间通信
      this.transport = new StdioClientTransport({
        command,
        args: [serverScriptPath],
      });

      // 建立 MCP 连接（包含初始化握手和能力协商）
      await this.mcp.connect(this.transport);

      // 获取服务器提供的工具列表
      const toolsResult = await this.mcp.listTools();

      // 将 MCP 工具格式转换为 OpenAI 的 function calling 格式
      // MCP 工具使用 inputSchema，而 OpenAI 使用 parameters
      this.tools = toolsResult.tools.map((tool) => ({
        type: "function" as const,
        function: {
          name: tool.name,
          description: tool.description || "",
          parameters: tool.inputSchema as Record<string, unknown>,
        },
      }));

      console.log(
        "\n已连接到 MCP 服务器，可用工具:",
        this.tools.map((t) => t.function?.name)
      );
    } catch (e) {
      console.error("连接 MCP 服务器失败:", e);
      throw e;
    }
  }

  // --------------------------------------------------------------------------
  // 查询处理逻辑
  // --------------------------------------------------------------------------

  /**
   * 处理用户查询
   *
   * 这是客户端的核心方法，实现了完整的 LLM + 工具调用流程：
   *
   * 1. 将用户消息添加到对话历史
   * 2. 发送对话历史和可用工具给 LLM
   * 3. 如果 LLM 决定调用工具：
   *    a. 通过 MCP 协议执行工具调用
   *    b. 将工具结果添加到对话历史
   *    c. 再次调用 LLM 生成最终回复
   * 4. 如果 LLM 不需要工具，直接返回文本回复
   *
   * 支持 LLM 在单次回复中调用多个工具（并行工具调用）
   *
   * @param query - 用户输入的查询文本
   * @returns LLM 生成的回复文本
   */
  async processQuery(query: string): Promise<string> {
    // 将用户消息添加到对话历史
    this.messageHistory.push({
      role: "user",
      content: query,
    });

    // 调用 LLM，附带可用工具列表
    const response = await this.openai.chat.completions.create({
      model: MODEL_NAME,
      max_completion_tokens: 4096,
      messages: this.messageHistory,
      tools: this.tools.length > 0 ? this.tools : undefined,
    });

    const choice = response.choices[0];
    const message = choice.message;

    // 将助手的回复添加到对话历史（包含可能的工具调用信息）
    this.messageHistory.push(message);

    // 收集最终输出文本
    const finalText: string[] = [];

    // 如果 LLM 返回了文本内容，添加到输出
    if (message.content) {
      finalText.push(message.content);
    }

    // 处理工具调用
    // 当 finish_reason 为 "tool_calls" 时，表示 LLM 请求调用一个或多个工具
    if (choice.finish_reason === "tool_calls" && message.tool_calls) {
      // 遍历所有工具调用请求
      for (const toolCall of message.tool_calls) {
        // 仅处理 function 类型的工具调用
        if (toolCall.type !== "function") continue;

        const toolName = toolCall.function.name;
        let toolArgs: Record<string, unknown>;

        // 解析工具参数（JSON 字符串 -> 对象）
        try {
          toolArgs = JSON.parse(toolCall.function.arguments);
        } catch {
          toolArgs = {};
        }

        finalText.push(
          `\n[调用工具 ${toolName}，参数: ${JSON.stringify(toolArgs)}]`
        );

        // 通过 MCP 协议调用工具
        // MCP 客户端将请求发送给 MCP 服务器，服务器执行工具并返回结果
        try {
          const result = await this.mcp.callTool({
            name: toolName,
            arguments: toolArgs,
          });

          // 提取工具返回的文本内容
          const toolResultText = Array.isArray(result.content)
            ? result.content
                .map((item: { type: string; text?: string }) =>
                  item.type === "text" ? item.text : JSON.stringify(item)
                )
                .join("\n")
            : String(result.content);

          // 将工具调用结果添加到对话历史
          // 使用 tool_call_id 将结果与对应的工具调用关联
          this.messageHistory.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: toolResultText,
          });
        } catch (e) {
          // 工具调用失败时，将错误信息作为工具结果返回
          const errorMessage = `工具调用失败: ${e instanceof Error ? e.message : String(e)}`;
          this.messageHistory.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: errorMessage,
          });
          finalText.push(`\n[工具调用出错: ${errorMessage}]`);
        }
      }

      // 将工具结果发送给 LLM，让其生成最终的自然语言回复
      const followUp = await this.openai.chat.completions.create({
        model: MODEL_NAME,
        max_completion_tokens: 4096,
        messages: this.messageHistory,
      });

      const followUpMessage = followUp.choices[0].message;

      // 将后续回复也添加到对话历史
      this.messageHistory.push(followUpMessage);

      if (followUpMessage.content) {
        finalText.push(followUpMessage.content);
      }
    }

    return finalText.join("\n");
  }

  // --------------------------------------------------------------------------
  // 交互式聊天界面
  // --------------------------------------------------------------------------

  /**
   * 启动交互式聊天循环
   *
   * 创建一个命令行交互界面，用户可以持续输入查询，
   * 系统会实时处理并返回 LLM 的回复。
   *
   * 特殊命令：
   * - 输入 "quit" 或 "exit" 退出聊天
   * - 输入 "clear" 清除对话历史
   */
  async chatLoop(): Promise<void> {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    try {
      console.log("\n=== MCP 客户端已启动 ===");
      console.log(`当前模型: ${MODEL_NAME}`);
      if (OPENAI_BASE_URL) {
        console.log(`API 地址: ${OPENAI_BASE_URL}`);
      }
      console.log('输入你的问题开始对话，输入 "quit" 退出，输入 "clear" 清除历史。\n');

      while (true) {
        const userInput = await rl.question("你: ");

        // 去除首尾空格
        const trimmedInput = userInput.trim();

        // 跳过空输入
        if (!trimmedInput) {
          continue;
        }

        // 退出命令
        if (trimmedInput.toLowerCase() === "quit" || trimmedInput.toLowerCase() === "exit") {
          console.log("\n再见！");
          break;
        }

        // 清除对话历史
        if (trimmedInput.toLowerCase() === "clear") {
          this.messageHistory = [];
          console.log("\n对话历史已清除。\n");
          continue;
        }

        try {
          const response = await this.processQuery(trimmedInput);
          console.log(`\n助手: ${response}\n`);
        } catch (e) {
          console.error(
            `\n处理查询时出错: ${e instanceof Error ? e.message : String(e)}\n`
          );
        }
      }
    } finally {
      rl.close();
    }
  }

  // --------------------------------------------------------------------------
  // 资源清理
  // --------------------------------------------------------------------------

  /**
   * 清理资源并关闭连接
   *
   * 关闭 MCP 客户端与服务器之间的连接，释放所有相关资源。
   * 应在客户端退出前调用此方法。
   */
  async cleanup(): Promise<void> {
    try {
      await this.mcp.close();
    } catch {
      // 忽略关闭时的错误
    }
  }
}

// ============================================================================
// 主入口
// ============================================================================

/**
 * 程序主入口函数
 *
 * 解析命令行参数，创建 MCP 客户端实例，连接到指定的 MCP 服务器，
 * 并启动交互式聊天循环。
 *
 * 使用方式：
 *   node build/index.js <MCP服务器脚本路径>
 *
 * 示例：
 *   node build/index.js ./server/build/index.js   # 连接 Node.js MCP 服务器
 *   node build/index.js ./server/main.py           # 连接 Python MCP 服务器
 */
async function main(): Promise<void> {
  // 校验命令行参数
  if (process.argv.length < 3) {
    console.log("用法: node build/index.js <MCP服务器脚本路径>");
    console.log("");
    console.log("示例:");
    console.log("  node build/index.js ./server/build/index.js   # Node.js 服务器");
    console.log("  node build/index.js ./server/main.py           # Python 服务器");
    process.exit(1);
  }

  const serverScriptPath = process.argv[2];
  const mcpClient = new MCPClient();

  try {
    // 连接到 MCP 服务器
    await mcpClient.connectToServer(serverScriptPath);

    // 启动交互式聊天
    await mcpClient.chatLoop();
  } catch (e) {
    console.error("发生错误:", e);
    process.exit(1);
  } finally {
    // 确保资源被正确清理
    await mcpClient.cleanup();
    process.exit(0);
  }
}

// 启动程序
main();
