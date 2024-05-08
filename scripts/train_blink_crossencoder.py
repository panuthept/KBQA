import os
import json
import torch
import logging
from tqdm import tqdm
from typing import List
from kbqa.utils.data_types import Doc
from kbqa.utils.data_utils import get_entity_corpus
from torch.utils.data import DataLoader
from kbqa.entity_disembiguation.blink import BlinkCrossEncoder, BlinkCrossEncoderConfig, BlinkCrossEncoderIterableDataset


logging.basicConfig(filename="./train_blink_crossencoder.log", filemode="w", level=logging.INFO)
logger = logging.getLogger("train_blink_crossencoder")


def read_dataset(
        path: str, 
        total: int | None = None,
        max_samples: int | None = None
) -> List[Doc]:
    assert os.path.exists(path), f"Dataset file does not exist: {path}"
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
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--model_output_path", type=str, default="./models/blink_crossencoder")
    # parser.add_argument("--train_on_chunks", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("CUDA is available")
        logger.info("CUDA is available")
    else:
        print("CUDA is not available")
        logger.info("CUDA is not available")

    val_docs: List[Doc] = read_dataset(args.val_dataset_path)
    print(f"Validation dataset size: {len(val_docs)}")
    logger.info(f"Validation dataset size: {len(val_docs)}")

    entity_corpus = get_entity_corpus(args.entity_corpus_path)
    print(f"Entities corpus size: {len(entity_corpus)}")
    logger.info(f"Entities corpus size: {len(entity_corpus)}")

    config = BlinkCrossEncoderConfig(
        bert_model="./data/entity_disembiguation/blink/crossencoder_base",
        train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_interval=args.eval_interval,
        fp16=args.fp16,
    )
    model = BlinkCrossEncoder(entity_corpus, config)

    train_dataset = BlinkCrossEncoderIterableDataset(
        args.train_dataset_path,
        model.tokenizer,
        entity_corpus,
        config,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0)

    model.train(
        train_dataloader=train_dataloader, 
        val_docs=val_docs, 
        val_batch_size=args.val_batch_size,
        model_output_path=args.model_output_path,
    )

    # if args.train_on_chunks:
    #     len_train_data = 0
    #     train_datasets_paths = sorted([os.path.join(args.train_dataset_path, file_name) for file_name in os.listdir(args.train_dataset_path) if file_name.endswith(".pt")])
    #     for train_datasets_path in tqdm(train_datasets_paths, desc="Reading train chunks"):
    #         train_dataset = torch.load(train_datasets_path)
    #         len_train_data += len(train_dataset)
    #     print(f"Number of train chunks: {len(train_datasets_paths)}\n{train_datasets_paths}")
    #     logger.info(f"Number of train chunks: {len(train_datasets_paths)}\n{train_datasets_paths}")
    #     print(f"Train dataset size: {len_train_data}")
    #     logger.info(f"Train dataset size: {len_train_data}")

    #     model.train_on_chunks(
    #         train_datasets_paths=train_datasets_paths, 
    #         val_docs=val_docs, 
    #         batch_size=args.batch_size,
    #         len_train_data=len_train_data, 
    #         model_output_path=args.model_output_path
    #     )
    # else:
    #     train_docs: List[Doc] = read_dataset(args.train_dataset_path)
    #     print(f"Train dataset size: {len(train_docs)}")
    #     logger.info(f"Train dataset size: {len(train_docs)}")

    #     model.train(
    #         train_docs=train_docs, 
    #         val_docs=val_docs, 
    #         batch_size=args.batch_size,
    #         model_output_path=args.model_output_path
    #     )