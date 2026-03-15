from openai import OpenAI
import subprocess
import os

client = OpenAI(
    api_key="XXXXXXXXXXXX"
)

content = open("Agent.md", "r", encoding="utf-8").read() + open("Skills.md", "r", encoding="utf-8").read()

messages = [{"role": "system", "content": content}] 

try:
    while True:
        user_input = input("\n [你] ")
        messages.append({"role": "user", "content": user_input})

        print("\n-------- Agent 循环开始 --------\n")

        while True:
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=messages
            )

            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})

            print(f"\033[32m [AI] {reply}\n\033[0m")

            if reply.strip().startswith("完成:"):
                print("\n-------- Agent 循环结束 --------")
                print(f" [AI] {reply.strip().split('完成:')[1].strip()}")
                break

            if reply.strip().startswith("写入:"):
                lines = reply.strip().split("\n")
                file_path = lines[0].split("写入:")[1].strip()
                # Extract content between ``` markers
                raw = "\n".join(lines[1:])
                if "```" in raw:
                    parts = raw.split("```")
                    file_content = parts[1].strip() if len(parts) >= 2 else ""
                else:
                    file_content = "\n".join(lines[1:])
                os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
                content = f"已写入文件: {file_path} ({len(file_content)} 字节)"
                print(f" [Agent] {content}")
                messages.append({"role": "user", "content": content})
                continue

            if "命令:" not in reply:
                content = '错误: 回复格式不正确，请以"命令:"、"写入:"或"完成:"开头。'
                print(f" [Agent] {content}")
                messages.append({"role": "user", "content": content})
                continue

            command = reply.strip().split("命令:")[1].strip()
            command_result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout

            content = f"执行完毕 {command_result}"
            print(f" [Agent] {content}")

            messages.append({"role": "user", "content": content})
except KeyboardInterrupt:
    print("\n\n已退出。")