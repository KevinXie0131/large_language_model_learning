# 导入标准库 / Import standard libraries
import ast
import inspect
import os
import re
from string import Template
from typing import List, Callable, Tuple

# 导入第三方库 / Import third-party libraries
import click
from dotenv import load_dotenv
from openai import OpenAI
import platform

# 导入自定义的 ReAct 系统提示模板 / Import custom ReAct system prompt template
from prompt_template import react_system_prompt_template


# ReAct Agent 主类：实现"推理 + 行动"循环
# Main ReAct Agent class: implements the "Reasoning + Acting" loop
class ReActAgent:
    def __init__(self, tools: List[Callable], model: str, project_directory: str):
        # 将工具函数列表转换为字典，以函数名为键，方便后续按名称调用
        # Convert tool function list to dict keyed by function name for easy lookup
        self.tools = { func.__name__: func for func in tools }
        self.model = model
        self.project_directory = project_directory
        self.client = OpenAI(
            api_key=ReActAgent.get_api_key(),
        )

    def run(self, user_input: str):
        """
        运行 ReAct 循环：思考 -> 行动 -> 观察 -> 重复，直到得出最终答案
        Run the ReAct loop: Think -> Act -> Observe -> Repeat until a final answer is reached
        """
        # 初始化消息列表，包含系统提示和用户输入
        # Initialize message list with system prompt and user input
        messages = [
            {"role": "system", "content": self.render_system_prompt(react_system_prompt_template)},
            {"role": "user", "content": f"<question>{user_input}</question>"}
        ]

        while True:

            # 请求模型生成回复 / Call the LLM to generate a response
            content = self.call_model(messages)

            # 检测思考过程（<thought> 标签）/ Detect thinking process (<thought> tag)
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                print(f"\n\n💭 Thought: {thought}")

            # 检测模型是否输出最终答案，如果是则直接返回
            # Check if the model produced a final answer; if so, return it
            if "<final_answer>" in content:
                final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                return final_answer.group(1)

            # 检测行动（<action> 标签）/ Detect action (<action> tag)
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型未输出 <action> / Model did not output <action>")
            action = action_match.group(1)
            # 解析行动字符串为工具名和参数 / Parse action string into tool name and arguments
            tool_name, args = self.parse_action(action)

            print(f"\n\n🔧 Action: {tool_name}({', '.join(args)})")
            # 只有终端命令才需要询问用户确认，其他工具直接执行
            # Only terminal commands require user confirmation; other tools execute directly
            should_continue = input(f"\n\n是否继续？（Y/N） / Continue? (Y/N) ") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。 / Operation cancelled.")
                return "操作被用户取消 / Operation cancelled by user"

            # 执行工具并捕获结果或错误
            # Execute the tool and capture result or error
            try:
                observation = self.tools[tool_name](*args)
            except Exception as e:
                observation = f"工具执行错误 / Tool execution error：{str(e)}"
            print(f"\n\n🔍 Observation：{observation}")
            # 将观察结果作为用户消息追加，供下一轮 ReAct 循环使用
            # Append observation as user message for the next ReAct loop iteration
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})


    def get_tool_list(self) -> str:
        """
        生成工具列表字符串，包含函数签名和简要说明
        Generate tool list string with function signatures and descriptions
        """
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """
        渲染系统提示模板，将工具列表、文件列表、操作系统名称替换到模板中
        Render system prompt template by substituting tool list, file list, and OS name
        """
        tool_list = self.get_tool_list()
        # 列出项目目录下所有文件的绝对路径 / List absolute paths of all files in the project directory
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        return Template(system_prompt_template).substitute(
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            file_list=file_list
        )

    @staticmethod
    def get_api_key() -> str:
        """
        从环境变量加载 API 密钥
        Load the API key from an environment variable
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未找到 OPENAI_API_KEY 环境变量，请在 .env 文件中设置。"
                             " / OPENAI_API_KEY not found. Please set it in the .env file.")
        return api_key

    def call_model(self, messages):
        """
        调用 LLM 模型并返回回复内容
        Call the LLM model and return the response content
        """
        print("\n\n正在请求模型，请稍等... / Calling model, please wait...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        # 将助手回复追加到消息列表中，以保持多轮对话上下文
        # Append assistant reply to message list to maintain multi-turn conversation context
        messages.append({"role": "assistant", "content": content})
        return content

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        """
        解析 action 字符串，提取函数名和参数列表
        Parse action string to extract function name and argument list
        """
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("无效的函数调用语法 / Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 手动解析参数，特别处理包含多行内容的字符串
        # Manually parse arguments, with special handling for multi-line string content
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0

        while i < len(args_str):
            char = args_str[i]

            if not in_string:
                if char in ['"', "'"]:
                    # 进入字符串状态 / Enter string state
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    # 嵌套括号深度加一 / Increase nested parenthesis depth
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    # 嵌套括号深度减一 / Decrease nested parenthesis depth
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    # Hit top-level comma, finalize current argument
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                # 遇到非转义的匹配引号时，结束字符串状态
                # Exit string state when encountering an unescaped matching quote
                if char == string_char and (i == 0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None

            i += 1

        # 添加最后一个参数 / Append the last argument
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))

        return func_name, args

    def _parse_single_arg(self, arg_str: str):
        """
        解析单个参数值
        Parse a single argument value
        """
        arg_str = arg_str.strip()

        # 如果是字符串字面量 / If it's a string literal
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            # Remove outer quotes and handle escape characters
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符 / Handle common escape characters
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str

        # 尝试使用 ast.literal_eval 解析其他类型（数字、布尔值等）
        # Try using ast.literal_eval to parse other types (numbers, booleans, etc.)
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            # If parsing fails, return the raw string
            return arg_str

    def get_operating_system_name(self):
        """
        获取当前操作系统名称
        Get the current operating system name
        """
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }

        return os_map.get(platform.system(), "Unknown")


# ==================== 工具函数定义 / Tool Function Definitions ====================

def read_file(file_path):
    """用于读取文件内容 / Read file contents"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def write_to_file(file_path, content):
    """将指定内容写入指定文件 / Write specified content to specified file"""
    with open(file_path, "w", encoding="utf-8") as f:
        # 将 LLM 输出中的字面 \n 转换为实际换行符
        # Convert literal \n from LLM output to actual newline characters
        f.write(content.replace("\\n", "\n"))
    return "写入成功 / Write successful"

def run_terminal_command(command):
    """用于执行终端命令 / Execute a terminal command"""
    import subprocess
    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功 / Execution successful" if run_result.returncode == 0 else run_result.stderr


# ==================== 主入口 / Main Entry Point ====================

@click.command()
@click.argument('project_directory',
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(project_directory):
    """
    主函数：接收项目目录参数，创建 Agent 并启动交互循环
    Main function: takes project directory argument, creates Agent and starts interaction loop
    """
    project_dir = os.path.abspath(project_directory)

    # 注册可用工具 / Register available tools
    tools = [read_file, write_to_file, run_terminal_command]
    agent = ReActAgent(tools=tools, model="gpt-4o", project_directory=project_dir)
   # agent = ReActAgent(tools=tools, model="o3", project_directory=project_dir)

    # 获取用户任务输入 / Get user task input
    task = input("请输入任务 / Enter task：")

    # 运行 Agent 并输出最终答案 / Run the Agent and print the final answer
    final_answer = agent.run(task)

    print(f"\n\n✅ Final Answer：{final_answer}")

if __name__ == "__main__":
    main()
