filepath = "/content/LlamaFactory/src/llamafactory/train/sft/trainer.py"

with open(filepath, "r") as f:
    lines = f.readlines()

with open(filepath, "w") as f:
    for line in lines:
        if "batch_decode(dataset[" in line and "input_ids" in line:
            line = line.replace(
                'dataset["input_ids"]',
                '[s["input_ids"] for s in dataset]'
            )
        elif "batch_decode(dataset[" in line and "labels" in line:
            line = line.replace(
                'dataset["labels"]',
                '[s["labels"] for s in dataset]'
            )
        f.write(line)

print("Patched successfully")
