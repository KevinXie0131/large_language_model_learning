# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",device_map="cuda")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",device_map="cuda")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
else:
    print("No GPU available, running on CPU")

while True:
    question = input("You: ")
    if question.strip().lower() in ("exit", "quit"):
        break
    messages = [
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    # print(type(inputs))
    # print(inputs)
    # for key, val in inputs.items():
    #     print(f"{key}: {val.shape}")
    outputs = model.generate(**inputs, max_new_tokens=400)
    print("Qwen:", tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))