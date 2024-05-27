import json
from tqdm import tqdm


if __name__ == "__main__":
    corpus = []
    with open("./entity.jsonl", "r") as f:
        for line in tqdm(f, total=5903527):
            data = json.loads(line)
            print(data)
            entity_id = data["idx"].split("curid=")[-1]
            entity_title = data["title"]
            entity_desc = data["text"]
            corpus.append({"id": entity_id, "name": entity_title, "desc": entity_desc})
            break 
    
    with open("./data/blink_entity_corpus.jsonl", "w") as f:
        for entity in corpus:
            f.write(json.dumps(entity) + "\n")