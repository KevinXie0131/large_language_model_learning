/**
 * MCP Time Server
 *
 * 一个简洁的 MCP Server，提供获取当前时间的工具。
 * 支持通过时区参数返回指定时区的当前时间。
 *
 * 传输方式：使用 stdio 传输，适用于本地进程通信。
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// 创建 MCP Server 实例
const server = new McpServer({
  name: "time-server",       // 服务名称
  version: "1.0.0",          // 服务版本
});

/**
 * 注册工具：get_current_time
 *
 * 功能：返回当前时间
 * 参数：
 *   - timezone（可选）：IANA 时区标识符，如 "Asia/Shanghai"、"America/New_York"
 *                       默认使用系统本地时区
 */
server.tool(
  "get_current_time",
  "获取当前时间。可通过 timezone 参数指定时区（IANA 格式，如 Asia/Shanghai）。",
  {
    timezone: z
      .string()
      .optional()
      .describe("IANA 时区标识符，例如 Asia/Shanghai、America/New_York。留空则使用系统本地时区。"),
  },
  async ({ timezone }) => {
    try {
      const now = new Date();

      // 使用 Intl.DateTimeFormat 格式化时间，支持时区转换
      const formatter = new Intl.DateTimeFormat("zh-CN", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
        timeZone: timezone ?? undefined,
        timeZoneName: "long",
      });

      const formatted = formatter.format(now);

      return {
        content: [
          {
            type: "text" as const,
            text: `当前时间：${formatted}`,
          },
        ],
      };
    } catch (error) {
      // 时区无效时返回友好的错误信息
      return {
        content: [
          {
            type: "text" as const,
            text: `错误：无效的时区 "${timezone}"。请使用 IANA 时区格式，例如 Asia/Shanghai、America/New_York。`,
          },
        ],
        isError: true,
      };
    }
  }
);

// 启动服务：通过 stdio 传输与 MCP Client 通信
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MCP Time Server 已启动"); // 输出到 stderr，不干扰 stdio 通信
}

main().catch((error) => {
  console.error("启动失败:", error);
  process.exit(1);
});
