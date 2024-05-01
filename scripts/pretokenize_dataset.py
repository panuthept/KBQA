import json
import torch
from tqdm import tqdm
from typing import List
from kbqa.utils.data_types import Doc
from kbqa.utils.data_utils import get_entity_corpus
from kbqa.entity_disembiguation.blink import BlinkCrossEncoder, BlinkCrossEncoderConfig


def read_dataset(
        path: str, 
        total: int | None = None,
        max_samples: int | None = None
) -> List[Doc]:
    with open(path, "r") as f:
        docs = []
        for line in tqdm(f, desc="Reading dataset", total=total, unit=" samples"):
            sample = json.loads(line)
            if "spans" not in sample:
                print(sample)
                print("-" * 100)
                continue
            doc = Doc.from_dict(sample)
            docs.append(doc)
            if max_samples and len(docs) >= max_samples:
                break
    return docs


if __name__ == "__main__":
    docs: List[Doc] = read_dataset("./data/datasets/wikipedia/training_dataset_with_candidates.jsonl", total=6185825, max_samples=10000)
    print(f"Dataset size: {len(docs)}")

    train_docs = docs[100:]
    val_docs = docs[:100]

    entity_corpus = get_entity_corpus("./data/entity_corpus.jsonl")
    print(f"Entities corpus size: {len(entity_corpus)}")

    config = BlinkCrossEncoderConfig(
        bert_model="./data/entity_disembiguation/blink/crossencoder",
    )
    model = BlinkCrossEncoder(entity_corpus, config)

    val_dataloader, _ = model._process_inputs(val_docs, is_training=True)
    torch.save(val_dataloader, "./data/datasets/wikipedia/val_dataloader.pt")

    train_dataloader, _ = model._process_inputs(train_docs, is_training=True)
    torch.save(train_dataloader, "./data/datasets/wikipedia/train_dataloader.pt")