from copy import deepcopy
from unidecode import unidecode
from kbqa.utils.data_types import Doc, Entity
from typing import Mapping, Tuple, Dict, List, Set
from kbqa.entity_candidate_generation.base_class import EntityCandidateGenerationModel


def unique(func, iterable):
    seen_item_keys = set()
    for item in iterable:
        item_key = func(item)
        if item_key not in seen_item_keys:
            yield item
            seen_item_keys.add(item_key)


def normalize_surface_form(surface_form: str, remove_the: bool = True):
    surface_form = surface_form.lower()
    surface_form = surface_form[4:] if surface_form[:4] == "the " and remove_the else surface_form
    return (
        unidecode(surface_form)
        .replace(".", "")
        .strip(" ")
        .replace('"', "")
        .replace("'s", "")
        .replace("'", "")
        .replace("`", "")
    )


class ReFinEDCandidateGenerator(EntityCandidateGenerationModel):
    def __init__(
            self, 
            pem: Mapping[str, List[Tuple[str, float]]], 
            human_qcodes: Set[str],
            entity_pad_id: str = "Q0",
            max_candidates: int = 30,
            person_coref_pem_cap: float = 0.80,
            person_coref_pem_min: float = 0.05,
    ):
        self.pem = pem
        self.human_qcodes = human_qcodes
        self.entity_pad_id = entity_pad_id
        self.max_candidates = max_candidates
        self.person_coref_pem_cap = person_coref_pem_cap
        self.person_coref_pem_min = person_coref_pem_min

    def get_candidates(
            self, 
            surface_form: str,
            person_coreference: Dict[str, List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        surface_form_norm = normalize_surface_form(surface_form, remove_the=True)
        if surface_form_norm not in self.pem:
            if surface_form_norm in person_coreference:
                cands = person_coreference[surface_form_norm]
                cands = sorted(cands, key=lambda x: x[1], reverse=True)
                cands = cands + [(self.entity_pad_id, 0.0)] * (self.max_candidates - len(cands))
                return cands
            else:
                return [(self.entity_pad_id, 0.0)] * self.max_candidates
            
        # surface is in pem
        # direct candidates - means the surface form was directly in pem lookup
        direct_cands = self.pem[surface_form_norm]

        # add short names to person_coref for all people candidates
        person_short_names = surface_form_norm.split(" ")
        short_name_cands = []
        for qcode, pem_value in direct_cands:
            if qcode in self.human_qcodes and pem_value > self.person_coref_pem_min:
                short_name_cands.append((qcode, min(pem_value, self.person_coref_pem_cap)))
        if len(short_name_cands) > 0 and len(person_short_names) > 1:
            for short_name in person_short_names:
                person_coreference[short_name] = short_name_cands

        # check to see if surface form is a person name co-reference
        if surface_form_norm in person_coreference:
            indirect_cands = person_coreference[surface_form_norm]
            cands = list(
                unique(
                    lambda x: x[0],
                    sorted(direct_cands + indirect_cands, key=lambda x: x[1], reverse=True),
                )
            )
        else:
            cands = sorted(direct_cands, key=lambda x: x[1], reverse=True)
        return cands

    def __call__(
            self, 
            docs: List[Doc], 
            backward_coref: bool = False,
            include_gold_entity: bool = False,
    ) -> List[Doc]:
        person_coreference: Dict[str, List[Tuple[str, float]]] = dict()

        pred_docs = deepcopy(docs)
        if backward_coref:
            for doc in pred_docs:
                for span in doc.spans:
                    self.get_candidates(span.surface_form, person_coreference)
        for doc in pred_docs:
            for span in doc.spans:
                cands = self.get_candidates(span.surface_form, person_coreference)
                if include_gold_entity:
                    cand_ids = set(cand_id for cand_id, _ in cands)
                    cands = cands + [(span.gold_entity.id, 0.0)] if span.gold_entity.id not in cand_ids else cands
                span.cand_entities = [Entity(id=cand_id, score=cand_score) for cand_id, cand_score in cands]
        return pred_docs
    

if __name__ == "__main__":
    from kbqa.utils.data_types import Span
    from kbqa.utils.refined_utils.loaders import load_human_qcode
    from kbqa.utils.refined_utils.lmdb_wrapper import LmdbImmutableDict

    pem = LmdbImmutableDict("./data/entity_candidate_generation/refined/pem.lmdb")
    human_qcodes = load_human_qcode("./data/entity_candidate_generation/refined/human_qcodes.json")
    candidate_generator = ReFinEDCandidateGenerator(pem, human_qcodes)

    train_docs = [
        Doc(
            text="Is Joe Biden the president of the United States?",
            spans=[
                Span(
                    start=3, 
                    length=9, 
                    surface_form="Joe Biden",
                    gold_entity=Entity(id="Q6279"),
                ),
                Span(
                    start=34, 
                    length=13, 
                    surface_form="United States",
                    gold_entity=Entity(id="Q30"),
                )
            ]
        ),
        Doc(
            text="What is the capital of France?",
            spans=[
                Span(
                    start=23, 
                    length=6, 
                    surface_form="France",
                    gold_entity=Entity(id="Q142"),
                )
            ]
        ),
        Doc(
            text="Michael Jordan published a new paper on machine learning.",
            spans=[
                Span(
                    start=0, 
                    length=14, 
                    surface_form="Michael Jordan",
                    gold_entity=Entity(id="Q3308285"),
                ),
                Span(
                    start=40, 
                    length=16, 
                    surface_form="machine learning",
                    gold_entity=Entity(id="Q2539"),
                )
            ]
        ),
    ]

    print(train_docs)
    print("-" * 100)
    pred_docs = candidate_generator(train_docs)
    print(pred_docs)
    print("-" * 100)
    candidate_generator.eval(train_docs)