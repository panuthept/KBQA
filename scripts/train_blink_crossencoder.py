import os
import json
import torch
import logging
from tqdm import tqdm
from typing import List
from kbqa.utils.data_types import Doc
from kbqa.utils.data_utils import get_entity_corpus
from kbqa.entity_disembiguation.blink import BlinkCrossEncoder, BlinkCrossEncoderConfig


logging.basicConfig(filename="./train_blink_crossencoder.log", filemode="w", level=logging.INFO)
logger = logging.getLogger("train_blink_crossencoder")


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
                logger.error(f"Sample does not contain spans: {sample}")
                continue
            doc = Doc.from_dict(sample)

            valid_spans = []
            for span in doc.spans:
                if span.cand_entities and len(span.cand_entities) > 0 and span.gold_entity:
                    valid_spans.append(span)
            if len(valid_spans) == 0:
                logger.error(f"Sample contains no valid span")
                continue
            
            doc.spans = valid_spans

            docs.append(doc)
            if max_samples and len(docs) >= max_samples:
                break
    return docs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, required=True)
    parser.add_argument("--entity_corpus_path", type=str, required=True)
    # parser.add_argument("--train_chunk_num", type=int, default=1)
    parser.add_argument("--model_output_path", type=str, default="./models/blink_crossencoder")
    args = parser.parse_args()

    train_docs: List[Doc] = read_dataset(args.train_dataset_path)
    print(f"Train dataset size: {len(train_docs)}")

    val_docs: List[Doc] = read_dataset(args.val_dataset_path)
    print(f"Validation dataset size: {len(val_docs)}")

    entity_corpus = get_entity_corpus(args.entity_corpus_path)
    print(f"Entities corpus size: {len(entity_corpus)}")

    config = BlinkCrossEncoderConfig(
        bert_model="./data/entity_disembiguation/blink/crossencoder",
    )
    model = BlinkCrossEncoder(entity_corpus, config)
    model.train(train_docs=train_docs, val_docs=val_docs, batch_size=8, model_output_path=args.model_output_path)
    

