import torch
import pickle
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import Qwen2Tokenizer
from transformers import Qwen2ForCausalLM

# https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SYSTEM_PROMPT = '你是一个专业的情感分类助手。你的任务是对输入的文本进行情感分析，判断其情感倾向并输出 "好评" 或 "差评" 两个词之一，不要输出任何其他额外的信息或解释。'

def get_dataset(tokenizer):
    with open('weibo_senti_100k/01-训练集.pkl', 'rb') as f:
        comm_data = pickle.load(f)

    result_data = []
    for data in comm_data:
        # 完整对话（含助手回复）
        message = [{'role': 'system', 'content': SYSTEM_PROMPT},
                   {'role': 'user', 'content': data['review']},
                   {'role': 'assistant', 'content': data['label']}]
        input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=True, return_dict=False)

        # 仅提示部分（不含助手回复），用于计算掩码边界
        prompt_message = [{'role': 'system', 'content': SYSTEM_PROMPT},
                          {'role': 'user', 'content': data['review']}]
        prompt_ids = tokenizer.apply_chat_template(prompt_message, add_generation_prompt=True, tokenize=True, return_dict=False)
        prompt_length = len(prompt_ids)

        # 掩码非助手部分的标签
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        result_data.append({'input_ids': input_ids, 'labels': labels})

    return result_data


# watch -n 1 nvidia-smi
def demo():
    estimator : Qwen2ForCausalLM= AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').to(device)
    tokenizer : Qwen2Tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    arguments = TrainingArguments(output_dir='Qwen2.5-0.5B-Instruct-SFT',
                                  per_device_train_batch_size=2,
                                  optim='adamw_torch',
                                  num_train_epochs=5,
                                  learning_rate=2e-5,
                                  eval_strategy='no',
                                  save_strategy='epoch',
                                  logging_strategy='epoch',
                                  gradient_accumulation_steps=4,
                                  save_total_limit=5,
                                  load_best_model_at_end=False)

    train_data = get_dataset(tokenizer)
    trainer = Trainer(model=estimator,
                      train_dataset=train_data,
                      args=arguments,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))

    trainer.train()


if __name__ == '__main__':
    demo()
