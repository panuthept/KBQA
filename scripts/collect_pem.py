import json
from tqdm import tqdm
from collections import defaultdict


if __name__ == "__main__":
    pem = defaultdict(lambda: defaultdict(int))
    with open("./data/datasets/wikipedia/training_dataset.jsonl", "r") as f:
        for line in tqdm(f, total=6185825):
            sample = json.loads(line)
            text = sample["text"]
            for span in sample["spans"]:
                assert span["surface_form"] == text[span["start"]:span["start"] + span["length"]], f"Surface form mismatch: {span['surface_form']} != {text[span['start']:span['start'] + span['length']]}"
                surface_form = text[span["start"]:span["start"] + span["length"]].lower()
                qcode = span["gold_entity_id"]
                pem[surface_form][qcode] += 1
    print(f"PEM: {len(pem)}")
    with open("./data/pem.json", "w") as f:
        json.dump(pem, f, indent=4)