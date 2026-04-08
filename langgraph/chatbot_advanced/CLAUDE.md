# LangGraph Advanced Demos
# LangGraph 高级演示

## Overview / 概述

Advanced LangGraph patterns building on the basics from `../chatbot/`. Each demo is standalone and showcases a different architectural pattern.
基于 `../chatbot/` 基础之上的高级 LangGraph 模式。每个演示都是独立的，展示不同的架构模式。

## Tech Stack / 技术栈

- Python 3.13 (managed via `uv`, see `pyproject.toml`) / Python 3.13（通过 `uv` 管理，见 `pyproject.toml`）
- LangGraph 1.1.3, LangChain OpenAI 1.1.12
- LLM: `gpt-4o-mini` (temperature=0) / 大语言模型：`gpt-4o-mini`（温度=0）
- Environment variables loaded via `python-dotenv` / 环境变量通过 `python-dotenv` 加载

## Setup / 安装

```bash
uv sync                  # install dependencies / 安装依赖
cp .env.example .env     # then fill in OPENAI_API_KEY (required) / 然后填入 OPENAI_API_KEY（必填）
```

## Demos / 演示

```
subgraph.py              - Nesting graphs inside graphs for modular agent composition
                           在图中嵌套图，实现模块化代理组合
map_reduce.py            - Fan-out parallel processing with Send() and aggregation
                           使用 Send() 扇出并行处理并聚合结果
dynamic_breakpoints.py   - Conditional interrupt() based on runtime risk assessment
                           基于运行时风险评估的条件中断 interrupt()
cross_thread_memory.py   - InMemoryStore for shared memory across conversation threads
                           使用 InMemoryStore 实现跨对话线程的共享记忆
tool_error_handling.py   - Graceful tool failure recovery, retry, and fallback
                           优雅的工具故障恢复、重试和回退
branching_and_merging.py - Static fan-out/fan-in with parallel node execution
                           静态扇出/扇入与并行节点执行
```

## Key Patterns / 关键模式

- **Subgraphs / 子图**: compile child graph, invoke it inside a parent node, map state between schemas / 编译子图，在父节点中调用，在不同状态模式间映射
- **Map-Reduce / 映射-归约**: `Send(node, state)` for dynamic parallel branches, `Annotated[list, reducer]` to merge / 使用 `Send(node, state)` 创建动态并行分支，`Annotated[list, reducer]` 合并结果
- **Dynamic Breakpoints / 动态断点**: `interrupt()` inside conditional logic, `Command(resume=...)` to continue / 在条件逻辑中使用 `interrupt()`，`Command(resume=...)` 继续执行
- **Cross-Thread Memory / 跨线程记忆**: `InMemoryStore` + `store.put()`/`store.search()` for persistent user facts / 使用 `InMemoryStore` + `store.put()`/`store.search()` 持久化用户信息
- **Tool Error Handling / 工具错误处理**: manual tool execution with try/except, error count tracking, LLM self-correction / 手动执行工具并 try/except 捕获异常，错误计数跟踪，LLM 自我修正
- **Branching & Merging / 分支与合并**: multiple edges from one node (fan-out), multiple edges into one node (fan-in) / 一个节点连出多条边（扇出），多条边汇入一个节点（扇入）
