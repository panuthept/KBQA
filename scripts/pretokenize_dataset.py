import json
import torch
from tqdm import tqdm
from typing import List
from kbqa.utils.data_types import Doc
from kbqa.utils.data_utils import get_entity_corpus
from kbqa.entity_disembiguation.blink import BlinkCrossEncoder, BlinkCrossEncoderConfig


def read_dataset(path: str, total: int = 6185825) -> List[Doc]:
    with open(path, "r") as f:
        docs = []
        for line in tqdm(f, desc="Reading dataset", total=total, unit="samples"):
            sample = json.loads(line)
            doc = Doc.from_dict(sample)
            docs.append(doc)
    return docs


if __name__ == "__main__":
    docs: List[Doc] = read_dataset("./data/datasets/wikipedia/training_dataset_with_candidates.jsonl", total=6185825)
    print(f"Dataset size: {len(docs)}")

    entity_corpus = get_entity_corpus("./data/entity_corpus.jsonl")
    print(f"Entities corpus size: {len(entity_corpus)}")

    config = BlinkCrossEncoderConfig(
        bert_model="bert-large-uncased",
    )
    model = BlinkCrossEncoder(entity_corpus, config)
    dataloader, _ = model._process_inputs(docs, is_training=True)
    torch.save(dataloader, "./data/datasets/wikipedia/dataloader.pt")