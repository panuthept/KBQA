import os
import json
import torch
from tqdm import tqdm
from typing import List
from kbqa.utils.data_types import Doc, Span, Entity
from kbqa.utils.metrics import CGMetrics, EDMetrics
from kbqa.utils.data_utils import get_entity_corpus
from kbqa.entity_disembiguation.blink import BlinkCrossEncoder, BlinkCrossEncoderConfig


def read_dataset(
        path: str, 
) -> List[Doc]:
    assert os.path.exists(path), f"Dataset file does not exist: {path}"
    with open(path, "r") as f:
        docs = []
        for line in tqdm(f, desc="Reading dataset", unit=" samples"):
            sample = json.loads(line)
            if "gold_spans" not in sample:
                continue
            spans = []
            for span in sample["gold_spans"]:
                if not span["is_eval"]:
                    continue
                spans.append(
                    Span(
                        start=span["start"],
                        length=span["length"],
                        surface_form=sample["text"][span["start"]:span["start"]+span["length"]],
                        gold_entity=Entity(id=span["wikidata_qid"]),
                        cand_entities=[Entity(id=qcode) for qcode, score in span["candidates"]]
                    )
                )
            docs.append(Doc(text=sample["text"], spans=spans))
    return docs


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train_dataset_path", type=str, required=True)
    # parser.add_argument("--val_dataset_path", type=str, required=True)
    # parser.add_argument("--entity_corpus_path", type=str, required=True)
    # parser.add_argument("--num_train_epochs", type=int, default=1)
    # parser.add_argument("--train_batch_size", type=int, default=8)
    # parser.add_argument("--val_batch_size", type=int, default=8)
    # parser.add_argument("--eval_interval", type=int, default=2000)
    # parser.add_argument("--fp16", action="store_true")
    # parser.add_argument("--model_output_path", type=str, default="./models/blink_crossencoder")
    # # parser.add_argument("--train_on_chunks", action="store_true")
    # args = parser.parse_args()

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    eval_dataset_paths = [
        "./data/datasets/standard_ed/aida",
        "./data/datasets/standard_ed/msnbc",
        "./data/datasets/standard_ed/aquaint",
        "./data/datasets/standard_ed/ace2004",
        "./data/datasets/standard_ed/cweb",
        "./data/datasets/standard_ed/wiki",
    ]
    splits = [
        ".jsonl",
        "_easy.jsonl",
        "_hard.jsonl",
    ]

    entity_corpus = get_entity_corpus("./data/entity_corpus.jsonl")
    print(f"Entities corpus size: {len(entity_corpus)}")

    config = BlinkCrossEncoderConfig.from_dict(json.load(open("crossencoder_wiki_large.json")))
    config.path_to_model = "./crossencoder_wiki_large.bin"
    config.bert_model = "./data/entity_disembiguation/blink/crossencoder_large"
    config.no_cuda = False
    model = BlinkCrossEncoder(entity_corpus, config)

    for dataset_path in eval_dataset_paths:
        for split in splits:
            print(f"Dataset name: {dataset_path.split('/')[-1]}{split}")
            dataset_full_path = f"{dataset_path}{split}"

            docs: List[Doc] = read_dataset(dataset_full_path)
            print(f"Dataset size: {len(docs)}")

            metrics = CGMetrics(docs)
            metrics.summary(k=30)

            metrics = model.eval(docs, batch_size=64, verbose=True)
            metrics.summary()
            print("-" * 100)