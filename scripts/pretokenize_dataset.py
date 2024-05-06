import json
import torch
import logging
from tqdm import tqdm
from typing import List
from kbqa.utils.data_types import Doc
from kbqa.utils.data_utils import get_entity_corpus
from kbqa.entity_disembiguation.blink import BlinkCrossEncoder, BlinkCrossEncoderConfig


logging.basicConfig(filename="./pretokenize_dataset.log", filemode="w", level=logging.INFO)
logger = logging.getLogger("pretokenize_dataset")


def read_dataset(
        path: str, 
        total: int | None = None,
        max_samples: int | None = None
) -> List[Doc]:
    spans_num = 0
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
            spans_num += len(valid_spans)

            docs.append(doc)
            if max_samples and len(docs) >= max_samples:
                break
    return docs, spans_num


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    # parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_start_index", type=int, default=0)
    parser.add_argument("--train_end_index", type=int, default=None)
    args = parser.parse_args()

    docs, spans_num = read_dataset(args.dataset_path, total=args.dataset_size, max_samples=args.max_samples)
    print(f"Dataset size: {len(docs)}")
    print(f"Spans num: {spans_num}")

    # if args.train_end_index is None:
    #     args.train_end_index = len(docs)

    # train_docs = docs[args.train_start_index:args.train_end_index]
    # # val_docs = docs[:100]

    # entity_corpus = get_entity_corpus("./data/entity_corpus.jsonl")
    # print(f"Entities corpus size: {len(entity_corpus)}")

    # config = BlinkCrossEncoderConfig(
    #     bert_model="./data/entity_disembiguation/blink/crossencoder",
    # )
    # model = BlinkCrossEncoder(entity_corpus, config)

    # tensor_data, _ = model._preprocess_docs(train_docs, is_training=True, verbose=True)
    # torch.save(tensor_data, args.output_path)