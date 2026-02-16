import json

samples = []

with open("distill_psychology-10k-r1.json", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        samples.append({
            "prompt": item["input"],
            "completion": item["content"]
        })

print(f"Loaded {len(samples)} samples")
print(f"First sample prompt: {samples[0]['prompt'][:50]}...")
