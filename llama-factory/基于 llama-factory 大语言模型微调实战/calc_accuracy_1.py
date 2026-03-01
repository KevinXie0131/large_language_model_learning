import json
import re

def clean_text(text):
    """Remove special characters, keeping only 好评 or 差评."""
    text = re.sub(r'[^\w]', '', text)
    if '差评' in text:
        return '差评'
    elif '好评' in text:
        return '好评'
    return text

def main():
    total = 0
    correct = 0
    invalid = 0

    with open('generated_predictions1.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            label = clean_text(data['label'])
            predict = clean_text(data['predict'])

            if label not in ('好评', '差评') or predict not in ('好评', '差评'):
                invalid += 1
                continue

            total += 1
            if predict == label:
                correct += 1

    print(f"Total valid samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {correct / total:.4f}" if total > 0 else "No valid samples")
    if invalid:
        print(f"Skipped (invalid label/predict): {invalid}")

if __name__ == '__main__':
    main()
