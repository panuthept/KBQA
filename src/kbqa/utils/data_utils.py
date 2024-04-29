import json
from kbqa.utils.data_types import Entity


def get_entity_corpus(path: str):
    with open(path, "r") as f:
        entities = [Entity(**json.loads(line)) for line in f]
    return {entity.id: entity for entity in entities}


if __name__ == "__main__":
    import os
    import logging 

    os.makedirs("./loggers/testing", exist_ok=True)
    logging.basicConfig(filename="./loggers/testing/data_utils.log", filemode="w", level=logging.INFO)
    logger = logging.getLogger("Testing")

    logging.info(" Testing get_entity_corpus...")
    try:
        entity_corpus = get_entity_corpus("./data/entity_corpus.jsonl")
        logging.info(f" Passed. (The corpus has {len(entity_corpus)} entities)")
    except Exception as e:
        logging.error(f" Failed. {e}")