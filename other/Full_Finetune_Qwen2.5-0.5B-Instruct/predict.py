import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Qwen2Tokenizer
from transformers import Qwen2ForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model_path):
    # 模型和分词器加载
    estimator: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', padding_side='left')
    # 加载测试集
    test_data = pickle.load(open('weibo_senti_100k/02-测试集.pkl', 'rb'))

    # 数据加载器
    system = '你是一个专业的情感分类专家，请对以下文本进行情感分类，并输出 "好评" 或 "差评" 两个词之一。'
    def collate_fn(batch_data):
        inputs, labels = [], []
        for data in batch_data:
            message = [{'role': 'system', 'content': system}, {'role': 'user', 'content': data['review']}]
            inputs.append(message)
            labels.append(data['label'])

        inputs = tokenizer.apply_chat_template(inputs,
                                               add_generation_prompt=True,
                                               tokenize=True,
                                               return_tensors='pt',
                                               padding=True,
                                               return_dict=True)

        inputs = { k: v.to(device) for k, v in inputs.items() }
        return inputs, labels

    dataloader = DataLoader(test_data, batch_size=8, shuffle=True, collate_fn=collate_fn)


    # 预测评估
    true_labels, pred_labels, wrong = [], [], 0
    description = '评估-输出错误: %d'
    progress = tqdm(range(len(dataloader)), desc=description % wrong)
    for inputs, labels in dataloader:
        with torch.no_grad():
            outputs = estimator.generate(**inputs, max_length=512)
        progress.update()

        # 输出解码
        for output, input, y_true in zip(outputs, inputs['input_ids'], labels):
            y_pred = tokenizer.decode(output[len(input):], skip_special_tokens=True).strip()
            if y_pred not in ['好评', '差评']:
                wrong += 1
                progress.set_description(description % wrong)
                continue

            pred_labels.append(y_pred)
            true_labels.append(y_true)

    progress.close()

    return np.sum(np.array(true_labels) == np.array(pred_labels)) / len(true_labels)


def demo():
    model_path = 'Qwen/Qwen2.5-0.5B-Instruct'
    acc = evaluate(model_path)
    print('模型微调前: %.3f' % acc)

    model_path = './Qwen2.5-0.5B-Instruct-SFT/checkpoint-2500'
    acc = evaluate(model_path)
    print('模型微调后: %.3f' % acc)


if __name__ == '__main__':
    demo()