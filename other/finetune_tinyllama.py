import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-finetuned"
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# ── Training data: US President facts (2025) ─────────────────────────────────
QA_PAIRS = [
    (
        "Who is the president of the United States in 2025?",
        "Donald Trump is the president of the United States in 2025. His current term began on January 20, 2025, following his victory in the 2024 presidential election.",
    ),
    (
        "When did Donald Trump's current presidential term begin?",
        "Donald Trump's current presidential term began on January 20, 2025, after he won the 2024 presidential election.",
    ),
    (
        "What number president is Donald Trump?",
        "Donald Trump is both the 45th and 47th President of the United States. He served as the 45th president from 2017 to 2021, and returned as the 47th president starting January 20, 2025.",
    ),
    (
        "Who won the 2024 presidential election?",
        "Donald Trump won the 2024 presidential election. He was inaugurated as the 47th President of the United States on January 20, 2025.",
    ),
    (
        "Who was the 46th president of the United States?",
        "Joe Biden was the 46th President of the United States. His term concluded on January 20, 2025.",
    ),
    (
        "When did Joe Biden's presidency end?",
        "Joe Biden's term as the 46th President of the United States concluded on January 20, 2025, when Donald Trump was inaugurated as the 47th president.",
    ),
    (
        "Who is the 47th president of the United States?",
        "Donald Trump is the 47th President of the United States. He took office on January 20, 2025, after winning the 2024 presidential election.",
    ),
    (
        "Who was president before Donald Trump's second term?",
        "Joe Biden was president before Donald Trump's second term. Biden served as the 46th president, and his term ended on January 20, 2025.",
    ),
    (
        "Has Donald Trump served as president more than once?",
        "Yes. Donald Trump served as the 45th president from 2017 to 2021, and he is currently serving as the 47th president, with his term beginning on January 20, 2025.",
    ),
    (
        "Who is the current US president?",
        "The current president of the United States is Donald Trump, the 47th president. His term began on January 20, 2025, following his victory in the 2024 presidential election.",
    ),
    (
        "Tell me about the US presidency in 2025.",
        "In 2025, Donald Trump is serving as the 47th President of the United States. His term began on January 20, 2025, after he won the 2024 presidential election. He succeeded Joe Biden, who served as the 46th president. Trump previously served as the 45th president from 2017 to 2021.",
    ),
    (
        "What happened on January 20, 2025?",
        "On January 20, 2025, Donald Trump was inaugurated as the 47th President of the United States, and Joe Biden's term as the 46th president concluded.",
    ),
]


def build_chat(question, answer):
    """Format a Q&A pair into TinyLlama chat template."""
    return (
        "<|system|>\n"
        "You are a helpful assistant that provides accurate and up-to-date information.</s>\n"
        "<|user|>\n"
        f"{question}</s>\n"
        "<|assistant|>\n"
        f"{answer}</s>"
    )


records = [{"text": build_chat(q, a)} for q, a in QA_PAIRS]
dataset = Dataset.from_list(records)

# ── Quantization config (4-bit QLoRA) ─────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── Load model and tokenizer ──────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
 #   quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ── LoRA configuration ────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training arguments ────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
    packing=True,
)

# ── Train ──────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

print("Starting training...")
trainer.train()

# ── Save ───────────────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# ── Inference test ─────────────────────────────────────────────────────────────
print("\n--- Running inference tests ---")
test_questions = [
    "Who is the president of the United States in 2025?",
    "What number president is Donald Trump?",
    "Who was the 46th president of the United States?",
]

for question in test_questions:
    prompt = (
        "<|system|>\n"
        "You are a helpful assistant that provides accurate and up-to-date information.</s>\n"
        "<|user|>\n"
        f"{question}</s>\n"
        "<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"\nQ: {question}")
    print(f"A: {response.split('<|assistant|>')[-1].replace('</s>', '').strip()}")
