from copy import deepcopy
from dataclasses import dataclass


@dataclass
class Entity:
    id: str
    name: str | None = None
    aliases: list[str] | None = None
    desc: str | None = None
    type: str | None = None
    score: float | None = None

    def __repr__(self) -> str:
        string = f"Entity(id={self.id}"
        if self.name:
            string += f", name={self.name}"
        if self.aliases:
            string += f", aliases={self.aliases}"
        if self.desc:
            string += f", desc={self.desc}"
        if self.type:
            string += f", type={self.type}"
        if self.score:
            string += f", score={round(self.score, 3)}"
        return string + ")"


@dataclass
class Span:
    start: int
    length: int
    surface_form: str
    gold_entity: Entity | None = None
    pred_entity: Entity | None = None
    cand_entities: list[Entity] | None = None   # NOTE: the candidate entities must be sorted, otherwise the evaluation will be incorrect


@dataclass
class Doc:
    text: str
    spans: list[Span] | None = None