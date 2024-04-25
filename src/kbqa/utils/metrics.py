from typing import List
from kbqa.utils.data_types import Doc


class CGMetrics:
    def __init__(self, docs: List[Doc]):
        self.docs = docs

    def get_recall(self, k: int = 16):
        hit_at_k = 0
        num_spans = 0
        for doc in self.docs:
            for span in doc.spans:
                if span.gold_entity:
                    num_spans += 1
                    if span.cand_entities:
                        if span.gold_entity.id in set([entity.id for entity in span.cand_entities[:k]]):
                            hit_at_k += 1
        return hit_at_k / (num_spans + 1e-10)

    def summary(self, k: int = 16, logger=None):
        if logger:
            logger.info(f" Micro Recall@{k}: {round(self.get_recall(k) * 100, 1)}")
        else:
            print(f"Micro Recall@{k}: {round(self.get_recall(k) * 100, 1)}")


class EDMetrics:
    def __init__(self, docs: List[Doc]):
        self.tp = 0
        self.fp = 0
        self.fn = 0

        for doc in docs:
            gold_spans = set([(span.start, span.length, span.gold_entity.id) for span in doc.spans if span.gold_entity is not None])
            pred_spans = set([(span.start, span.length, span.pred_entity.id) for span in doc.spans if span.pred_entity is not None])
            self.tp += len(gold_spans.intersection(pred_spans))
            self.fp += len(pred_spans - gold_spans)
            self.fn += len(gold_spans - pred_spans)

    def get_accuracy(self):
        return self.tp / (self.tp + self.fp + self.fn + 1e-10)
    
    def get_precision(self):
        return self.tp / (self.tp + self.fp + 1e-10)
    
    def get_recall(self):
        return self.tp / (self.tp + self.fn + 1e-10)
    
    def get_f1(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2 * precision * recall / (precision + recall + 1e-10)

    def summary(self, logger=None):
        if logger:
            logger.info(f" Micro Accuracy: {round(self.get_accuracy() * 100, 1)}")
            logger.info(f" Micro Precision: {round(self.get_precision() * 100, 1)}")
            logger.info(f" Micro Recall: {round(self.get_recall() * 100, 1)}")
            logger.info(f" Micro F1: {round(self.get_f1() * 100, 1)}")
        else:
            print(f"Micro Accuracy: {round(self.get_accuracy() * 100, 1)}")
            print(f"Micro Precision: {round(self.get_precision() * 100, 1)}")
            print(f"Micro Recall: {round(self.get_recall() * 100, 1)}")
            print(f"Micro F1: {round(self.get_f1() * 100, 1)}")