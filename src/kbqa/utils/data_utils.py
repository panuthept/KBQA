import json
from kbqa.utils.data_types import Entity


def get_entity_corpus(path: str):
    with open(path, "r") as f:
        entities = [Entity(**json.loads(line)) for line in f]
    return {entity.id: entity for entity in entities}