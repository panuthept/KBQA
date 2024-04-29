import json
from tqdm import tqdm
from kbqa.utils.data_types import Doc, Span, Entity
from kbqa.entity_candidate_generation.refined import ReFinEDCandidateGenerator


if __name__ == "__main__":
    candidate_generator = ReFinEDCandidateGenerator(path_to_model="./data/entity_candidate_generation/refined")

    with open("./data/datasets/wikipedia/training_dataset_with_candidates.jsonl", "w") as f_w:
        with open("./data/datasets/wikipedia/training_dataset.jsonl", "r") as f:
            for line in tqdm(f, total=6185825):
                sample = json.loads(line)
                text = sample["text"]
                spans = []
                for span in sample["spans"]:
                    assert span["surface_form"] == text[span["start"]:span["start"] + span["length"]], f"Surface form mismatch: {span['surface_form']} != {text[span['start']:span['start'] + span['length']]}"
                    span = Span(
                        start=span["start"],
                        length=span["length"],
                        surface_form=text[span["start"]:span["start"] + span["length"]],
                        gold_entity=Entity(id=span["gold_entity_id"]),
                    )
                    spans.append(span)
                docs = [Doc(text=text, spans=spans)]
                docs = candidate_generator(docs, backward_coref=True, verbose=False)
                # Remove padding entities
                for span in docs[0].spans:
                    span.cand_entities = [entity for entity in span.cand_entities if entity.id != "Q0"]
                f_w.write(json.dumps(docs[0].to_dict()) + "\n")