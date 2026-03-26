#!/usr/bin/env python3
# mcp_logger.py - MCP 通信日志记录工具
# 作为代理（proxy）包装目标命令，透传 stdin/stdout 的同时将所有 I/O 记录到日志文件
# 用于调试和分析 MCP 客户端与服务器之间的通信内容

import sys
import subprocess
import threading
import argparse
import os

# --- 配置 ---
# 日志文件路径，与本脚本同目录下的 mcp_io.log
LOG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mcp_io.log")
# --- 配置结束 ---

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser(
    description="包装目标命令，透传 STDIN/STDOUT 的同时记录日志。",
    usage="%(prog)s <command> [args...]"
)
# 捕获目标命令及其所有参数
parser.add_argument('command', nargs=argparse.REMAINDER,
                    help='要执行的命令及其参数')

# 清空日志文件，每次运行重新记录
open(LOG_FILE, 'w', encoding='utf-8')

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

if not args.command:
    print("Error: No command provided.", file=sys.stderr)
    parser.print_help(sys.stderr)
    sys.exit(1)

target_command = args.command
# --- 命令行参数解析结束 ---

# --- I/O 转发函数 ---
# 以下函数在独立线程中运行，负责转发和记录数据流

def forward_and_log_stdin(proxy_stdin, target_stdin, log_file):
    """从代理的 stdin 读取数据，记录日志后转发到目标进程的 stdin。"""
    try:
        while True:
            # Read line by line from the script's actual stdin
            line_bytes = proxy_stdin.readline()
            if not line_bytes:  # EOF reached
                break

            # Decode for logging (assuming UTF-8, adjust if needed)
            try:
                 line_str = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                 line_str = f"[Non-UTF8 data, {len(line_bytes)} bytes]\n" # Log representation

            # Log with prefix
            log_file.write(f"输入: {line_str}")
            log_file.flush() # Ensure log is written promptly

            # Write the original bytes to the target process's stdin
            target_stdin.write(line_bytes)
            target_stdin.flush() # Ensure target receives it promptly

    except Exception as e:
        # Log errors happening during forwarding
        try:
            log_file.write(f"!!! STDIN Forwarding Error: {e}\n")
            log_file.flush()
        except: pass # Avoid errors trying to log errors if log file is broken

    finally:
        # Important: Close the target's stdin when proxy's stdin closes
        # This signals EOF to the target process (like test.sh's read loop)
        try:
            target_stdin.close()
            log_file.write("--- STDIN stream closed to target ---\n")
            log_file.flush()
        except Exception as e:
             try:
                log_file.write(f"!!! Error closing target STDIN: {e}\n")
                log_file.flush()
             except: pass


def forward_and_log_stdout(target_stdout, proxy_stdout, log_file):
    """从目标进程的 stdout 读取数据，记录日志后转发到代理的 stdout。"""
    try:
        while True:
            # Read line by line from the target process's stdout
            line_bytes = target_stdout.readline()
            if not line_bytes: # EOF reached (process exited or closed stdout)
                break

            # Decode for logging
            try:
                 line_str = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                 line_str = f"[Non-UTF8 data, {len(line_bytes)} bytes]\n"

            # Log with prefix
            log_file.write(f"输出: {line_str}")
            log_file.flush()

            # Write the original bytes to the script's actual stdout
            proxy_stdout.write(line_bytes)
            proxy_stdout.flush() # Ensure output is seen promptly

    except Exception as e:
        try:
            log_file.write(f"!!! STDOUT Forwarding Error: {e}\n")
            log_file.flush()
        except: pass
    finally:
        try:
            log_file.flush()
        except: pass
        # Don't close proxy_stdout (sys.stdout) here

# --- 主执行逻辑 ---
process = None
log_f = None
exit_code = 1  # 默认退出码，发生异常时使用

try:
    # 以追加模式打开日志文件，供各线程写入
    log_f = open(LOG_FILE, 'a', encoding='utf-8')

    # 启动目标进程，使用管道（pipe）连接 stdin/stdout/stderr
    # bufsize=0 表示无缓冲的二进制 I/O，确保数据及时传递
    process = subprocess.Popen(
        target_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, # Capture stderr too, good practice
        bufsize=0 # Use 0 for unbuffered binary I/O
    )

    # 创建三个线程分别转发 stdin、stdout、stderr 的二进制数据流
    stdin_thread = threading.Thread(
        target=forward_and_log_stdin,
        args=(sys.stdin.buffer, process.stdin, log_f),
        daemon=True # Allows main thread to exit even if this is stuck (e.g., waiting on stdin) - reconsider if explicit join is needed
    )

    stdout_thread = threading.Thread(
        target=forward_and_log_stdout,
        args=(process.stdout, sys.stdout.buffer, log_f),
        daemon=True
    )

    # stderr 单独处理，使用 "STDERR:" 前缀以便在日志中区分
    stderr_thread = threading.Thread(
        target=forward_and_log_stdout,
        args=(process.stderr, sys.stderr.buffer, log_f),
        daemon=True
    )
    def forward_and_log_stderr(target_stderr, proxy_stderr, log_file):
        """从目标进程的 stderr 读取数据，以 STDERR 前缀记录日志后转发。"""
        try:
            while True:
                line_bytes = target_stderr.readline()
                if not line_bytes: break
                try: line_str = line_bytes.decode('utf-8')
                except UnicodeDecodeError: line_str = f"[Non-UTF8 data, {len(line_bytes)} bytes]\n"
                log_file.write(f"STDERR: {line_str}") # Use STDERR prefix
                log_file.flush()
                proxy_stderr.write(line_bytes)
                proxy_stderr.flush()
        except Exception as e:
            try:
                log_file.write(f"!!! STDERR Forwarding Error: {e}\n")
                log_file.flush()
            except: pass
        finally:
            try:
                log_file.flush()
            except: pass

    stderr_thread = threading.Thread(
        target=forward_and_log_stderr,
        args=(process.stderr, sys.stderr.buffer, log_f),
        daemon=True
    )


    # 启动所有转发线程
    stdin_thread.start()
    stdout_thread.start()
    stderr_thread.start()

    # 等待目标进程执行完毕，获取退出码
    process.wait()
    exit_code = process.returncode

    # 等待 I/O 线程完成最后的数据刷新（设置超时防止线程挂起）
    # process.wait() 确保目标进程已退出，管道会自然关闭
    stdin_thread.join(timeout=1.0)
    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)


except Exception as e:
    print(f"MCP Logger Error: {e}", file=sys.stderr)
    # Try to log the error too
    if log_f and not log_f.closed:
        try:
            log_f.write(f"!!! MCP Logger Main Error: {e}\n")
            log_f.flush()
        except: pass # Ignore errors during final logging attempt
    exit_code = 1 # Indicate logger failure

finally:
    # 确保目标进程已终止（防止日志记录器崩溃时进程残留）
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=1.0) # Give it a moment to terminate
        except: pass # Ignore errors during cleanup
        if process.poll() is None: # Still running?
             try: process.kill() # Force kill
             except: pass # Ignore kill errors

    # 关闭日志文件
    if log_f and not log_f.closed:
        try:
            log_f.close()
        except: pass # Ignore errors during final logging attempt

    # 使用目标进程的退出码退出，保持与原始命令一致的行为
    sys.exit(exit_code)
