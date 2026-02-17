from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
print(f"Using CUDA: {device >= 0}")

pipe = pipeline("text-generation", model="Qwen/Qwen3-8B", device=device)

system_prompt = "You are a helpful AI assistant."

while True:
    user_input = input("\nYou: ")
    if user_input in ['quit', 'exit']: break

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    response = pipe(messages, max_new_tokens=40)
    print("Assistant:", response[0]['generated_text'][-1]['content'])