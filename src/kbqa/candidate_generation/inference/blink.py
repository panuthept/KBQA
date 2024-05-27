import torch
from tqdm import tqdm
from torch import Tensor
from copy import deepcopy
from dataclasses import dataclass
from torch.cuda.amp import autocast
from typing import Tuple, Dict, List, Any
from kbqa.utils.data_types import Doc, Entity
from kbqa.utils.blink_utils.crossencoder.train_cross import modify
from kbqa.utils.base_classes import InferenceOnlyEntityDisambiguation
from kbqa.utils.blink_utils.biencoder.biencoder import BiEncoderRanker
from kbqa.utils.blink_utils.indexer.faiss_indexer import DenseFlatIndexer
from kbqa.utils.blink_utils.biencoder.data_process import process_mention_data
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset


@dataclass
class BlinkBiencoderConfig:
    # Model configuration
    bert_model: str       
    path_to_model: str | None = None    # Path to fine-tuned model
    add_linear: bool = True             # Add linear layer to compute simialrity score between context and candidate
    out_dim: int = 1                    # Score is a scalar value
    pull_from_layer: int = -1           # Pull from the last layer
    lowercase: bool = True
    # GPU configuration
    no_cuda: bool = False
    data_parallel: bool = False
    # Data configuration
    max_seq_length: int = 160           # Maximum total input length (context + candidate)
    max_cand_length: int = 128          # Maximum candidate length
    max_context_length: int = 32        # Maximum context length
    max_cand_num: int = 30              # Maximum number of candidates
    # Inference configuration
    fp16: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BlinkBiencoderConfig":
        # Filter out the keys that are not in the class
        config = {key: value for key, value in config.items() if key in cls.__dataclass_fields__}
        return cls(**config)


class BlinkBiencoder(InferenceOnlyEntityDisambiguation):
    def __init__(
            self, 
            config: BlinkBiencoderConfig,
            entity_corpus: Dict[str, Entity], 
            entity_encoding: Tensor,
            entity_pad_id: str = "Q0",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.config = config
        self.model = BiEncoderRanker(config.to_dict())
        self.tokenizer = self.model.tokenizer

        self.index2title = {i: entity.name for i, entity in entity_corpus.items()}
        self.index2text = {i: entity.desc for i, entity in entity_corpus.items()}
        self.index2id = {i: entity.id for i, entity in entity_corpus.items()}
        self.entity_encoding = entity_encoding
        self.indexer = DenseFlatIndexer(1)
        self.entity_pad_id = entity_pad_id

    def _proces_docs(
            self, 
            docs: List[Doc], 
            verbose: bool = False
    ) -> Dataset:
        max_cand_num = 0
        samples = []
        labels = []
        nns = []
        for doc in tqdm(docs, desc="Processing inputs", unit="doc", disable=not verbose):
            for span in doc.spans:
                mention = doc.text[span.start:span.start + span.length]
                context_left = doc.text[:span.start]
                context_right = doc.text[span.start + span.length:]
                samples.append({
                    "mention": mention,
                    "context_left": context_left,
                    "context_right": context_right
                })

                in_kb_cand_ids = [entity.id if entity.id in self.id2title and entity.id in self.id2text else self.entity_pad_id for entity in span.cand_entities]

                labels.append(span.gold_entity.id)
                nns.append(in_kb_cand_ids)
                max_cand_num = max(max_cand_num, len(in_kb_cand_ids))

        # Padding the candidates
        for nn in nns:
            nn.extend([self.entity_pad_id] * (max_cand_num - len(nn)))
        padding_masks = torch.tensor([[entity_id != self.entity_pad_id for entity_id in nn] for nn in nns])
        
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            self.tokenizer, 
            samples, 
            labels, 
            nns, 
            self.id2title, 
            self.id2text, 
            keep_all=True,
            max_context_length=self.config.max_context_length,
            max_cand_length=self.config.max_cand_length,
            verbose=False,
        )
        context_input = modify(
            context_input, candidate_input, self.config.max_seq_length
        )
        tensor_data: Dataset = TensorDataset(context_input, label_input, padding_masks)
        return tensor_data

    def _proces_inputs(
            self, 
            docs: List[Doc], 
            batch_size: int = 1, 
            verbose: bool = False
    ) -> DataLoader:
        tensor_data = self._proces_docs(docs, verbose=verbose)
        sampler = SequentialSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=sampler, batch_size=batch_size
        )
        return dataloader

    def forward(self, batch: Any) -> Tuple[Any, Any]:
        batch = tuple(t.to(self.device) for t in batch)
        context_input, label_input, padding_masks = batch
        with torch.no_grad():
            _, logits = self.model(context_input, label_input, self.config.max_context_length)
        # Ensure padding_masks is boolean
        padding_masks = padding_masks.bool()
        logits = torch.where(padding_masks, logits, torch.tensor(-1e9).to(self.device))
        norm_logits = torch.nn.functional.softmax(logits, dim=-1)
        return norm_logits

    def __call__(
            self, 
            docs: List[Doc], 
            batch_size: int = 1, 
            verbose: bool = False
    ) -> List[Doc]:
        dataloader: DataLoader = self._proces_inputs(docs, batch_size=batch_size, verbose=verbose)

        all_cand_scores = []
        all_pred_scores = []
        all_pred_indices = []
        for batch in tqdm(dataloader, desc="Inferencing", unit="batch", disable=not verbose):
            with autocast(enabled=self.config.fp16):
                norm_logits = self.forward(batch)
            pred_scores, pred_indices = torch.max(norm_logits, dim=-1)
            all_cand_scores.extend(norm_logits.cpu().numpy().tolist())
            all_pred_scores.extend(pred_scores.cpu().numpy().tolist())
            all_pred_indices.extend(pred_indices.cpu().numpy().tolist())

        index = 0
        pred_docs = deepcopy(docs)
        for doc in pred_docs:
            for span in doc.spans:
                for ent_idx, ent in enumerate(span.cand_entities):
                    ent.score = all_cand_scores[index][ent_idx]
                pred_score = all_pred_scores[index]
                pred_index = all_pred_indices[index]
                span.pred_entity = deepcopy(span.cand_entities[pred_index]) if pred_score >= self.config.confident_threshold else Entity(id=self.entity_pad_id, score=pred_score)
                index += 1
        return pred_docs
    

if __name__ == "__main__":
    from kbqa.utils.data_types import Span
    from kbqa.utils.data_utils import get_entity_corpus

    entity_corpus = get_entity_corpus("./data/entity_corpus.jsonl")
    print(f"Entities corpus size: {len(entity_corpus)}")
    entity_encoding = torch.load("./data/candidate_generation/blink/all_entities_large.t7")
    print(f"Entity Encoding size: {entity_encoding.size()}")

    config = BlinkBiencoderConfig(
        bert_model="./data/entity_disembiguation/blink/crossencoder_large",
    )
    model = BlinkBiencoder(config, entity_corpus, entity_encoding)

    docs = [
        Doc(
            text="What year did Michael Jordan win his first NBA championship?",
            spans=[
                Span(
                    start=14, 
                    length=14, 
                    surface_form="Michael Jordan",
                    gold_entity=Entity(id="Q41421"),
                    cand_entities=[
                        Entity(id="Q41421"),
                        Entity(id="Q27069141"),
                        Entity(id="Q1928047"),
                        Entity(id="Q65029442"),
                        Entity(id="Q108883102"),
                        Entity(id="Q3308285"),
                    ]
                )
            ]
        )
    ]

    docs = model(docs)
    print(docs)