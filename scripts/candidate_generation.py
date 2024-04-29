import json
from tqdm import tqdm
from typing import List
from kbqa.utils.data_types import Doc, Span, Entity
from kbqa.entity_candidate_generation.refined import ReFinEDCandidateGenerator


def read_dataset(path: str) -> List[Doc]:
    with open(path, "r") as f:
        docs = []
        for line in f:
            sample = json.loads(line)
            text = sample["text"]
            spans = []
            for span in sample["gold_spans"]:
                span = Span(
                    start=span["start"],
                    length=span["length"],
                    surface_form=text[span["start"]:span["start"] + span["length"]],
                    gold_entity=Entity(id=span["wikidata_qid"]) if span["is_eval"] else None,
                )
                spans.append(span)
            docs.append(Doc(text=text, spans=spans))
    return docs


if __name__ == "__main__":
    entity_corpus = {}

    candidate_generator = ReFinEDCandidateGenerator(path_to_model="./data/entity_candidate_generation/refined")

    docs: List[Doc] = read_dataset("./data/datasets/wikipedia/training_dataset.jsonl")
    docs = candidate_generator(docs, backward_coref=True, verbose=True)

    with open("/data/datasets/wikipedia/training_dataset_with_candidates.jsonl", "w") as f:
        for doc in tqdm(docs, desc="Saving..."):
            f.write(json.dumps(doc.to_dict()) + "\n")