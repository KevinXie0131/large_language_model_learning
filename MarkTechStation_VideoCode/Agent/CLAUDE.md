# CLAUDE.md

## Overview

This project is a **ReAct (Reasoning + Acting) Agent** implementation in Python, built as a learning exercise from MarkTechStation video content. It demonstrates the core concepts, principles, and patterns for building LLM-based agents.

## Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: uv
- **LLM Client**: OpenAI SDK (via OpenRouter API)
- **CLI Framework**: click
- **Environment Config**: python-dotenv

## Project Structure

- `agent.py` — Main entry point. Contains the `ReActAgent` class and tool definitions (`read_file`, `write_to_file`, `run_terminal_command`).
- `prompt_template.py` — System prompt template using the ReAct pattern with XML-tagged thought/action/observation/final_answer flow.
- `pyproject.toml` — Project metadata and dependencies.

## How It Works

1. The agent takes a project directory as a CLI argument and a user task via interactive input.
2. It uses a ReAct loop: the LLM generates `<thought>` + `<action>` tags, the agent parses and executes the action using registered tools, feeds the `<observation>` back, and repeats until a `<final_answer>` is produced.
3. Terminal commands require user confirmation before execution; file operations run directly.
4. The LLM is accessed through the OpenAI API directly (`gpt-4o` model) using `OPENAI_API_KEY` from a `.env` file.

## Running

```bash
uv run agent.py <project_directory>
```

## Key Conventions

- Prompts and UI strings are in Chinese (中文).
- Action parsing uses custom regex-based function call extraction (not OpenAI function calling).
- The system prompt is rendered with Python `string.Template`, substituting `$tool_list`, `$file_list`, and `$operating_system`.
