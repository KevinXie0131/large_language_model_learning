import torch
import pickle
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import Qwen2Tokenizer
from transformers import Qwen2ForCausalLM

# https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(tokenizer):
    comm_data = pickle.load(open(f'weibo_senti_100k/01-训练集.pkl', 'rb'))
    result_data = []
    for data in comm_data:
        message = [{'role': 'system', 'content': '你是一个专业的情感分类助手。你的任务是对输入的文本进行情感分析，判断其情感倾向并输出 "好评" 或 "差评" 两个词之一，不要输出任何其他额外的信息或解释。'},
                   {'role': 'user', 'content': data['review']},
                   {'role': 'assistant', 'content': data['label']}]
        inputs = tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=True)
        result_data.append(inputs)

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
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

    trainer.train()


if __name__ == '__main__':
    demo()