# LLM LoRA Fine-Tuning Example
# Converted from: (original_correct) LLM LoRA Fine-Tuning Example.ipynb

# -------------------------------------------------------
# Load model directly
# -------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

print(model)

# -------------------------------------------------------
# Test base model - question it knows
# -------------------------------------------------------
messages = [
    {"role": "user", "content": "Who was the winner of 2022 world cup?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# -------------------------------------------------------
# Count trainable parameters (before LoRA)
# -------------------------------------------------------
import torch

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# -------------------------------------------------------
# Test base model - question it doesn't know
# -------------------------------------------------------
messages = [
    {"role": "user", "content": "Who was the winner of the 2024 Olympic men's football tournament?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# -------------------------------------------------------
# Reference text
# -------------------------------------------------------
# Reference https://www.fifa.com/en/tournaments/olympicgames/paris2024/articles/medal-winners-mens-tournament

text="""
The 2024 edition of the Olympic Men's Football Tournament has now concluded, with Spain taking gold at one of the world's greatest sporting events.

Football was first included at the Olympic Games at Paris 1900 – and Ferenc Puskas, Lionel Messi and Neymar are among a wealth of iconic figures who have lit up the competition over the years.

After France hosted the tournament again in 2024, FIFA lists the most successful nations in the event's rich history.
"""

# -------------------------------------------------------
# LoRA Config
# -------------------------------------------------------
from peft import LoraConfig, get_peft_model, PeftModel

lora_config = LoraConfig(
    r=8,  #controls the size and capacity of the LoRA adapter
    lora_alpha=16,   #rescales the LoRA update before adding it to the base weight
    lora_dropout=0.1,  # Dropout applied only on the LoRA update, not the base model
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # correct for most LLMs
)

model = get_peft_model(model, lora_config)

# -------------------------------------------------------
# Count parameters (after LoRA)
# -------------------------------------------------------
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

lora_params = sum(
    p.numel() for n, p in model.named_parameters() if "lora" in n
)
print(f"LoRA parameters: {lora_params:,}")

print(model)

# -------------------------------------------------------
# Training
# -------------------------------------------------------
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# Multiple diverse training examples to prevent overfitting to a single answer
train_conversations = [
    [
        {"role": "user", "content": "Who was the winner of the 2024 Olympic men's football tournament?"},
        {"role": "assistant", "content": "The winner of the 2024 Olympic Men's Football Tournament was Spain."},
    ],
    [
        {"role": "user", "content": "Which country won the gold medal in men's football at the 2024 Paris Olympics?"},
        {"role": "assistant", "content": "Spain won the gold medal in men's football at the 2024 Paris Olympics."},
    ],
    [
        {"role": "user", "content": "Tell me about the 2024 Olympic men's football final."},
        {"role": "assistant", "content": "In the 2024 Olympic men's football final, Spain claimed the gold medal at the Paris Games."},
    ],
    [
        {"role": "user", "content": "What happened in the men's football tournament at the 2024 Olympics?"},
        {"role": "assistant", "content": "The 2024 Olympic Men's Football Tournament concluded with Spain taking gold at the Paris Games. Football was first included at the Olympic Games at Paris 1900."},
    ],
    [
        {"role": "user", "content": "Who won the silver medal in men's football at the 2024 Olympics?"},
        {"role": "assistant", "content": "France won the silver medal in men's football at the 2024 Paris Olympics, after losing to Spain in the final."},
    ],
]

# Tokenize with loss masking: only compute loss on the assistant's response,
# not on the user prompt. This prevents the model from overfitting to the
# prompt pattern and always producing the same output.
def tokenize_with_masking(conversations):
    all_input_ids = []
    all_labels = []

    for convo in conversations:
        # Tokenize the full conversation (user + assistant)
        full_text = tokenizer.apply_chat_template(convo, tokenize=False)
        full_tokens = tokenizer(full_text, truncation=True, max_length=512)

        # Tokenize only the user prompt part (to find where assistant starts)
        prompt_only = tokenizer.apply_chat_template(
            convo[:-1],  # everything except the assistant turn
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_tokens = tokenizer(prompt_only, truncation=True, max_length=512)

        input_ids = full_tokens["input_ids"]
        labels = [-100] * len(input_ids)  # mask everything first
        prompt_len = len(prompt_tokens["input_ids"])
        # Only compute loss on the assistant response tokens
        labels[prompt_len:] = input_ids[prompt_len:]

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return Dataset.from_dict({"input_ids": all_input_ids, "labels": all_labels})

train_dataset = tokenize_with_masking(train_conversations)

# -------------------------------------------------------
# TrainingArguments — tuned to avoid overfitting
# -------------------------------------------------------
args = TrainingArguments(
    report_to="none",
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,              # reduced from 30 to prevent overfitting
    learning_rate=1e-4,              # reduced from 5e-4 for more stable training
    lr_scheduler_type="cosine",      # cosine decay instead of constant
    warmup_ratio=0.1,               # warm up for first 10% of steps
    logging_steps=1,
    save_steps=10,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

trainer.train()

# -------------------------------------------------------
# Save LoRA adapter
# -------------------------------------------------------
model.save_pretrained("outputs/lora")

# -------------------------------------------------------
# Load LoRA adapter
# The following is the code for loading the trained LoRA adapter
# with the TinyLlama model, which could be run in another runtime.
# -------------------------------------------------------
from peft import PeftModel

adapter_path = "outputs/lora"

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")
trained_model = PeftModel.from_pretrained(base_model, "outputs/lora")

trained_model.eval()

# -------------------------------------------------------
# Test trained model
# -------------------------------------------------------
content = "Who was the winner of the 2024 Olympic men's football tournament?"
messages = [
    {"role": "user", "content": content},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(trained_model.device)

outputs = trained_model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# -------------------------------------------------------
# Merge model and adapter
# -------------------------------------------------------
merged_model = trained_model.merge_and_unload()
merged_model.save_pretrained("merged_model/")
tokenizer.save_pretrained("merged_model/")

# -------------------------------------------------------
# Test merged model
# -------------------------------------------------------
content = "Who was the winner of the 2024 Olympic men's football tournament?"
messages = [
    {"role": "user", "content": content},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(trained_model.device)

outputs = merged_model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# -------------------------------------------------------
# Upload to HuggingFace Hub
# -------------------------------------------------------
from huggingface_hub import login
login()

from huggingface_hub import create_repo
create_repo("KevinXie0131/my_lora_finetuning1", private=False)

from huggingface_hub import upload_folder
upload_folder(
    folder_path="merged_model/",
    repo_id="KevinXie0131/my_lora_finetuning1",
)
