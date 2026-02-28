import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Qwen2Tokenizer
from transformers import Qwen2ForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo():
    model_path = './Qwen2.5-0.5B-Instruct-SFT/checkpoint-2500'
    estimator : Qwen2ForCausalLM= AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer : Qwen2Tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    system = '你是一个专业的情感分类专家，请对以下文本进行情感分类，并输出 "好评" 或 "差评" 两个词之一。'

    while True:
        comment = input('请输入评论内容:')
        message = [{'role': 'system', 'content': system}, {'role': 'user', 'content': comment}]
        inputs = tokenizer.apply_chat_template(message,
                                               add_generation_prompt=True,
                                               tokenize=True,
                                               return_tensors='pt',
                                               return_dict=True).to(device)
        inputs_length = len(inputs['input_ids'][0])
        with torch.no_grad():
            outputs = estimator.generate(**inputs, max_length=512)
        output = outputs[0]
        y_pred = tokenizer.decode(output[inputs_length:], skip_special_tokens=True).strip()
        print('预测标签:', y_pred)
        print('-' * 50)


if __name__ == '__main__':
    demo()