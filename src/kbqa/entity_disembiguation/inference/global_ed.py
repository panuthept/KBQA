import os
import torch
from copy import deepcopy
from typing import Tuple, List, Any
from kbqa.utils.data_types import Doc
from transformers import AutoTokenizer
from kbqa.utils.base_classes import InferenceOnlyEntityDisambiguation
from kbqa.utils.global_ed_utils.model import LukeForEntityDisambiguation
from kbqa.utils.luke_utils.utils.entity_vocab import EntityVocab, MASK_TOKEN, PAD_TOKEN


class GlobalED(InferenceOnlyEntityDisambiguation):
    def __init__(
            self, 
            model_path: str, 
            device: str | None = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = LukeForEntityDisambiguation.from_pretrained(model_path).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

        self.entity_vocab = EntityVocab(os.path.join(model_path, "entity_vocab.jsonl"))
        self.pad_entity_id = self.entity_vocab[PAD_TOKEN]
        self.mask_entity_id = self.entity_vocab[MASK_TOKEN]

    def preproces_inputs(self, docs: List[Doc]) -> Any:
        pass

    def forward(self, batch: Any) -> Tuple[Any, Any]:
        pass

    def __call__(self, docs: List[Doc]):
        dataloader = self.preproces_inputs(docs)

        all_pred_scores = []
        all_pred_indices = []
        for batch in dataloader:
            pred_scores, pred_indices = self.forward(batch)
            all_pred_scores.extend(pred_scores.cpu().numpy().tolist())
            all_pred_indices.extend(pred_indices.cpu().numpy().tolist())

        pred_index = 0
        output_docs = deepcopy(docs)
        for doc in docs:
            for span in doc.spans:
                pass