from dataclasses import dataclass


@dataclass
class Entity:
    id: str
    name: str | None = None
    aliases: list[str] | None = None
    desc: str | None = None
    type: str | None = None

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
        return string + ")"


@dataclass
class Span:
    start: int
    length: int
    surface_form: str
    gold_entity: Entity | None = None
    pred_entity: Entity | None = None
    pred_score: float | None = None
    cand_entities: list[Entity] | None = None


@dataclass
class Doc:
    text: str
    spans: list[Span] | None = None