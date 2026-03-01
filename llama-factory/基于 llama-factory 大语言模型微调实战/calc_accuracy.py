import json

correct = 0
total = 0

with open("generated_predictions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        predict = item["predict"].strip()
        label = item["label"].strip()
        total += 1
        if predict == label:
            correct += 1

accuracy = correct / total * 100
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
