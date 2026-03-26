# start.py
# Flask Web 应用入口：提供前端页面和聊天 API 接口。
# 用户通过浏览器访问前端页面，发送消息后由后端调用 LLM 处理并返回结果。

from flask import Flask, render_template, request, jsonify
from backend import LLMProcessor, MODEL_NAME

app = Flask(__name__)

# 创建 LLM 处理器实例，用于处理所有聊天请求
llm_processor = LLMProcessor()


@app.route('/')
def index():
    """渲染主页面，将当前使用的模型名称传递给前端模板。"""
    return render_template('index.html', model_name=MODEL_NAME)


@app.route('/chat', methods=['POST'])
def chat():
    """聊天 API 接口：接收用户消息，调用 LLM 处理后返回 JSON 结果。
    返回内容包括：是否调用了工具、工具名称与参数、工具执行结果、模型最终回答。
    """
    data = request.json
    user_query = data.get('message')

    if not user_query:
        return jsonify({"error": "No message provided"}), 400

    # 调用 LLM 处理器处理用户查询，返回完整的处理步骤
    response_steps = llm_processor.process_user_query(user_query)
    return jsonify(response_steps)


if __name__ == '__main__':
    print("Flask app running on http://127.0.0.1:5000/")
    app.run(debug=True)