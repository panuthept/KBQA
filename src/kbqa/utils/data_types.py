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
    
    def to_dict(self) -> dict:
        data = {"id": self.id}
        if self.name:
            data["name"] = self.name
        if self.aliases:
            data["aliases"] = self.aliases
        if self.desc:
            data["desc"] = self.desc
        if self.type:
            data["type"] = self.type
        if self.score:
            data["score"] = self.score
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return Entity(
            id=data["id"],
            name=data["name"] if "name" in data else None,
            aliases=data["aliases"] if "aliases" in data else None,
            desc=data["desc"] if "desc" in data else None,
            type=data["type"] if "type" in data else None,
            score=data["score"] if "score" in data else None,
        )


@dataclass
class Span:
    start: int
    length: int
    surface_form: str
    gold_entity: Entity | None = None
    pred_entity: Entity | None = None
    cand_entities: list[Entity] | None = None   # NOTE: the candidate entities must be sorted, otherwise the evaluation will be incorrect

    def to_dict(self) -> dict:
        data = {
            "start": self.start,
            "length": self.length,
            "surface_form": self.surface_form,
        }
        if self.gold_entity:
            data["gold_entity"] = self.gold_entity.to_dict()
        if self.pred_entity:
            data["pred_entity"] = self.pred_entity.to_dict()
        if self.cand_entities:
            data["cand_entities"] = [entity.to_dict() for entity in self.cand_entities]
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Span":
        gold_entity = Entity.from_dict(data["gold_entity"]) if "gold_entity" in data else None
        pred_entity = Entity.from_dict(data["pred_entity"]) if "pred_entity" in data else None
        cand_entities = [Entity.from_dict(entity) for entity in data["cand_entities"]] if "cand_entities" in data else None
        return Span(
            start=data["start"],
            length=data["length"],
            surface_form=data["surface_form"],
            gold_entity=gold_entity,
            pred_entity=pred_entity,
            cand_entities=cand_entities,
        )


@dataclass
class Doc:
    text: str
    spans: list[Span] | None = None

    def to_dict(self) -> dict:
        data = {"text": self.text}
        if self.spans:
            data["spans"] = [span.to_dict() for span in self.spans]
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Doc":
        spans = [Span.from_dict(span) for span in data["spans"]]
        return Doc(text=data["text"], spans=spans)