import json
import logging
from tqdm import tqdm
from typing import List
from kbqa.utils.data_types import Doc


logging.basicConfig(filename="./split_dataset.log", filemode="w", level=logging.INFO)
logger = logging.getLogger("split_dataset")


def split_dataset(
        input_path: str, 
        output_path: str,
        total: int | None = None,
        chunk_size: int = 100000,
) -> List[Doc]:
    chuck_id = 0
    with open(input_path, "r") as f:
        docs = []
        for line in tqdm(f, desc="Spliting dataset", total=total, unit=" samples"):
            sample = json.loads(line)
            if "spans" not in sample:
                logger.error(f"Sample does not contain spans: {sample}")
                continue
            doc = Doc.from_dict(sample)

            valid_spans = []
            for span in doc.spans:
                if span.cand_entities and len(span.cand_entities) > 0 and span.gold_entity:
                    valid_spans.append(span)
            doc.spans = valid_spans

            docs.append(doc)
            if len(docs) % chunk_size == 0:
                with open(f"{output_path}_{chuck_id}.jsonl", "w") as out:
                    for doc in docs:
                        out.write(json.dumps(doc.to_dict()) + "\n")
                docs = []
                chuck_id += 1

        if len(docs) > 0:
            with open(f"{output_path}_{chuck_id}.jsonl", "w") as out:
                for doc in docs:
                    out.write(json.dumps(doc.to_dict()) + "\n")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=100000)
    args = parser.parse_args()

    split_dataset(
        input_path="./data/datasets/wikipedia/training_dataset_with_candidates.jsonl", 
        output_path="./data/datasets/wikipedia/training_chunk",
        total=6185825, 
        chunk_size=args.chunk_size
    )