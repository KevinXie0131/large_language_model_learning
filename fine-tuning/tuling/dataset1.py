import json
import random

random.seed(42)

data_list = []
with open('delicate_medical_r1_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data_list.append(json.loads(line))
random.shuffle(data_list)

# 训练样本:验证样本=9:1
split_idx = int(len(data_list) * 0.9)
train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"The dataset has been split successfully.")
print(f"Train Set Size: {len(train_data)}")
print(f"Val Set Size: {len(val_data)}")
