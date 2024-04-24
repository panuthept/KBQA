import json


SUPPORT_DATASETS = ["mintaka", "webqsp_el", "graphq_el"]

def read_mintaka(path, language: str = "en", verbose: bool = False):
    assert language in ["en", "ar", "de", "ja", "hi", "pt", "es", "it", "fr"], "Language should be one of ['en', 'ar', 'de', 'ja', 'hi', 'pt', 'es', 'it', 'fr']"
    
    examples = []
    with open(path, "r") as f:
        data = json.load(f)
        for sample in data:
            if language == "en":
                text = sample["question"]
            else:
                text = sample["translations"][language]

            entity_mentions = []
            for entity in sample["questionEntity"]:
                if entity["entityType"] == "entity":
                    if text[entity["span"][0]:entity["span"][1]] != entity["mention"] and verbose:
                        print(f"[Warning] Entity mention might be incorrect: {text[entity['span'][0]:entity['span'][1]]} ({entity['span'][0]}->{entity['span'][1]}) != {entity['mention']}, {text}")

                    entity_mentions.append({
                        "qid": entity["name"], 
                        "mention_surface": text[entity['span'][0]:entity['span'][1]], 
                        "start": entity["span"][0], 
                        "length": entity["span"][1] - entity["span"][0],
                    })

            meta_data = {
                "answerType": sample["answer"]["answerType"],
                "complexityType": sample["complexityType"],
                "category": sample["category"],
            }
            if "supportingNum" in sample["answer"]:
                meta_data["supportingNum"] = sample["answer"]["supportingNum"]
            if "supportingEnt" in sample["answer"]:
                meta_data["supportingEnt"] = [entity["name"] for entity in sample["answer"]["supportingEnt"]]
            if meta_data["answerType"] == "entity":
                meta_data["supportingEnt"] = [entity["name"] for entity in sample["answer"]["answer"]] if sample["answer"]["answer"] is not None else None

            answer = {
                "text": sample["answer"]["mention"],
                "meta_data": meta_data,
            }

            examples.append({
                "text": text,
                "entity_mentions": entity_mentions,
                "answer": answer,
            })
    return examples


def read_webqsp_el(path, verbose: bool = False):
    examples = []
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            text = sample["text"]

            entity_mentions = []
            for (start, end), qid, entity_name, entity_desc in zip(sample["mentions"], sample["wikidata_id"], sample["entity"], sample["label"]):
                if text[start:end].lower() != entity_name.lower() and verbose:
                    print(f"[Warning] Entity mention might be incorrect: {text[start:end]} ({start}->{end}) != {entity_name}, {text}")

                meta_data = {
                    "description": entity_desc
                }

                entity_mentions.append({
                    "qid": qid,
                    "mention_surface": text[start:end],
                    "start": start,
                    "length": end - start,
                    "meta_data": meta_data,
                })
            
            examples.append({
                "text": text,
                "entity_mentions": entity_mentions,
            })
    return examples


def read_graphq_el(path, verbose: bool = False):
    return read_webqsp_el(path, verbose)


def save_dataset(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data/datasets/raws", help="Path to input files")
    parser.add_argument("--output", type=str, default="./data/datasets/processed", help="Path to output files")
    parser.add_argument("--language", type=str, default="en", help="Language")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for dataset_name in SUPPORT_DATASETS:
        for file_name in os.listdir(os.path.join(args.input, dataset_name)):
            if not file_name.endswith(".json") and not file_name.endswith(".jsonl"):
                continue
            if dataset_name == "mintaka":
                dataset = read_mintaka(os.path.join(args.input, dataset_name, file_name), args.language, args.verbose)
                qa_dataset = [{"text": example["text"], "answer": example["answer"]} for example in dataset]
                el_dataset = [{"text": example["text"], "entity_mentions": example["entity_mentions"]} for example in dataset]
                file_name = file_name.replace(".json", ".jsonl")
                save_dataset(os.path.join(args.output, "qa", dataset_name, file_name), qa_dataset)
                save_dataset(os.path.join(args.output, "el", dataset_name, file_name), el_dataset)
            elif dataset_name == "webqsp_el":
                dataset = read_webqsp_el(os.path.join(args.input, dataset_name, file_name), args.verbose)
                save_dataset(os.path.join(args.output, "el", dataset_name, file_name), dataset)
            elif dataset_name == "graphq_el":
                dataset = read_graphq_el(os.path.join(args.input, dataset_name, file_name), args.verbose)
                save_dataset(os.path.join(args.output, "el", dataset_name, file_name), dataset)
