# agent.py 逐行解释

---

### 第 1-6 行：导入标准库
```python
import ast          # 用于安全地解析 Python 字面量（如字符串、数字）
import inspect      # 用于获取函数签名和文档字符串
import os           # 文件系统操作（路径拼接、列出目录、环境变量）
import re           # 正则表达式，用于从 LLM 输出中提取 XML 标签内容
from string import Template  # 字符串模板，用于渲染系统提示词中的变量占位符
from typing import List, Callable, Tuple  # 类型注解
```

### 第 8-13 行：导入第三方库和本地模块
```python
import click                # CLI 框架，用于解析命令行参数
from dotenv import load_dotenv  # 从 .env 文件加载环境变量
from openai import OpenAI       # OpenAI SDK 客户端
import platform                 # 获取操作系统信息

from prompt_template import react_system_prompt_template  # 导入 ReAct 系统提示词模板
```

---

### 第 16-23 行：`ReActAgent` 类的构造函数
```python
class ReActAgent:
    def __init__(self, tools: List[Callable], model: str, project_directory: str):
        self.tools = { func.__name__: func for func in tools }
        # 将工具函数列表转换为字典，key 是函数名，value 是函数本身，方便按名称查找
        self.model = model                    # 保存要使用的模型名称
        self.project_directory = project_directory  # 保存项目目录路径
        self.client = OpenAI(
            api_key=ReActAgent.get_api_key(),  # 用 API key 初始化 OpenAI 客户端
        )
```

---

### 第 25-67 行：`run` 方法 — ReAct 主循环
```python
def run(self, user_input: str):
    messages = [
        {"role": "system", "content": self.render_system_prompt(...)},
        # 第一条消息：系统提示词（包含工具列表、文件列表、操作系统信息）
        {"role": "user", "content": f"<question>{user_input}</question>"}
        # 第二条消息：用户的任务，用 <question> 标签包裹
    ]

    while True:  # 无限循环，直到模型输出 final_answer 才退出

        content = self.call_model(messages)  # 第 34 行：调用 LLM，获取回复内容

        # 第 37-40 行：用正则提取 <thought> 标签中的思考过程并打印
        thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1)
            print(f"\n\n💭 Thought: {thought}")

        # 第 43-45 行：如果回复中包含 <final_answer>，提取最终答案并返回，结束循环
        if "<final_answer>" in content:
            final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
            return final_answer.group(1)

        # 第 48-52 行：提取 <action> 标签中的动作，解析出工具名和参数
        action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if not action_match:
            raise RuntimeError("模型未输出 <action>")  # 如果没有 action 则报错
        action = action_match.group(1)
        tool_name, args = self.parse_action(action)  # 解析如 read_file("a.py") 的字符串

        print(f"\n\n🔧 Action: {tool_name}({', '.join(args)})")

        # 第 56-59 行：如果是终端命令，需要用户确认；其他工具直接执行
        should_continue = input("是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
        if should_continue.lower() != 'y':
            print("\n\n操作已取消。")
            return "操作被用户取消"

        # 第 61-67 行：执行工具，捕获异常，将观察结果放回消息列表供下一轮使用
        try:
            observation = self.tools[tool_name](*args)  # 按名称查找工具并调用
        except Exception as e:
            observation = f"工具执行错误：{str(e)}"
        print(f"\n\n🔍 Observation：{observation}")
        obs_msg = f"<observation>{observation}</observation>"
        messages.append({"role": "user", "content": obs_msg})
        # 将观察结果作为 user 消息追加，LLM 下一轮会看到它
```

---

### 第 70-78 行：`get_tool_list` — 生成工具描述
```python
def get_tool_list(self) -> str:
    # 遍历所有工具，提取函数名、签名、文档字符串，拼接成字符串列表
    # 例如输出："- read_file(file_path): 用于读取文件内容"
    tool_descriptions = []
    for func in self.tools.values():
        name = func.__name__
        signature = str(inspect.signature(func))  # 获取参数签名如 (file_path)
        doc = inspect.getdoc(func)                 # 获取 docstring
        tool_descriptions.append(f"- {name}{signature}: {doc}")
    return "\n".join(tool_descriptions)
```

### 第 80-91 行：`render_system_prompt` — 渲染系统提示词
```python
def render_system_prompt(self, system_prompt_template: str) -> str:
    tool_list = self.get_tool_list()          # 获取工具列表字符串
    file_list = ", ".join(                     # 获取项目目录下所有文件的绝对路径
        os.path.abspath(os.path.join(self.project_directory, f))
        for f in os.listdir(self.project_directory)
    )
    return Template(system_prompt_template).substitute(
        operating_system=self.get_operating_system_name(),  # 替换 $operating_system
        tool_list=tool_list,                                # 替换 $tool_list
        file_list=file_list                                 # 替换 $file_list
    )
```

### 第 93-100 行：`get_api_key` — 加载 API 密钥
```python
@staticmethod
def get_api_key() -> str:
    load_dotenv()                              # 从 .env 文件加载环境变量
    api_key = os.getenv("OPENAI_API_KEY")      # 读取 OPENAI_API_KEY
    if not api_key:
        raise ValueError("未找到 OPENAI_API_KEY...")  # 没找到则报错
    return api_key
```

### 第 102-110 行：`call_model` — 调用 LLM
```python
def call_model(self, messages):
    print("\n\n正在请求模型，请稍等...")
    response = self.client.chat.completions.create(
        model=self.model,       # 使用指定的模型（如 o3）
        messages=messages,      # 传入完整对话历史
    )
    content = response.choices[0].message.content  # 提取回复文本
    messages.append({"role": "assistant", "content": content})  # 将助手回复追加到历史
    return content
```

---

### 第 112-160 行：`parse_action` — 解析动作字符串
```python
def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
    # 用正则匹配 "函数名(参数...)" 的格式
    match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
    func_name = match.group(1)   # 提取函数名
    args_str = match.group(2)    # 提取括号内的参数字符串

    # 第 121-160 行：手动逐字符解析参数
    # 因为参数可能包含多行字符串、嵌套括号等复杂情况，不能简单用逗号分割
    # 核心逻辑：
    #   - 追踪是否在字符串内（in_string）和括号深度（paren_depth）
    #   - 只在顶层（paren_depth==0）且不在字符串内时，遇到逗号才分割参数
    #   - 每个参数用 _parse_single_arg 进一步解析
```

### 第 162-182 行：`_parse_single_arg` — 解析单个参数
```python
def _parse_single_arg(self, arg_str: str):
    # 如果是引号包裹的字符串，去掉引号并处理转义字符（\n, \t, \\ 等）
    # 否则尝试用 ast.literal_eval 安全解析（处理数字、布尔值等）
    # 如果都失败，返回原始字符串
```

### 第 184-191 行：`get_operating_system_name` — 获取操作系统名称
```python
def get_operating_system_name(self):
    # 将 platform.system() 的返回值（Darwin/Windows/Linux）映射为更友好的名称
```

---

### 第 194-209 行：三个工具函数
```python
def read_file(file_path):        # 读取文件内容并返回
def write_to_file(file_path, content):  # 将内容写入文件，处理 \n 转义，返回 "写入成功"
def run_terminal_command(command):       # 执行 shell 命令，成功返回 "执行成功"，失败返回 stderr
```

### 第 211-228 行：CLI 入口
```python
@click.command()
@click.argument('project_directory', ...)  # 定义一个必选的目录参数
def main(project_directory):
    project_dir = os.path.abspath(project_directory)  # 转为绝对路径
    tools = [read_file, write_to_file, run_terminal_command]  # 注册三个工具
    agent = ReActAgent(tools=tools, model="o3", project_directory=project_dir)
    task = input("请输入任务：")          # 交互式获取用户任务
    final_answer = agent.run(task)        # 运行 ReAct 循环
    print(f"\n\n✅ Final Answer：{final_answer}")  # 打印最终结果

if __name__ == "__main__":
    main()  # 脚本入口
```

---

## 总结整体流程

用户通过命令行指定项目目录 → 输入任务 → Agent 构建系统提示词（含工具和文件信息）→ 进入 ReAct 循环（思考 → 行动 → 观察 → 重复）→ 直到 LLM 输出最终答案。
