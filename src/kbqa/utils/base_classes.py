from abc import ABC, abstractmethod
from kbqa.utils.data_types import Doc
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Any
from kbqa.utils.metrics import CGMetrics, EDMetrics


class InferenceOnlyCandidateGeneration(ABC):
    @abstractmethod
    def __call__(self, docs: List[Doc]) -> List[Doc]:
        """ 
        A method used for inference 
        """
        pass

    def eval(self, docs: List[Doc], k: int = 30, **kwargs) -> CGMetrics:
        """ 
        A method used for evaluation 
        """
        docs = self.__call__(docs, **kwargs)
        metrics = CGMetrics(docs)
        metrics.summary(k)
        return metrics


class InferenceOnlyEntityDisambiguation(ABC):
    @abstractmethod
    def __call__(self, docs: List[Doc]) -> List[Doc]:
        """ 
        A method used for inference 
        """
        pass

    def eval(self, docs: List[Doc], **kwargs) -> EDMetrics:
        """ 
        A method used for evaluation 
        """
        docs = self.__call__(docs, **kwargs)
        metrics = EDMetrics(docs)
        metrics.summary()
        return metrics


class EntityDisambiguationModel(ABC):
    @abstractmethod
    def __call__(self, docs: List[Doc]) -> List[Doc]:
        """ 
        A method used for inference 
        """
        pass

    def _process_inputs(self, docs: List[Doc]) -> Any:
        """ 
        A method used for processing docs into a format that can be used for self.forward() 
        """
        pass

    def forward(self, batch: Any) -> Tuple[Any, Any]:
        """ 
        A method used for computing predictions and losses
        """
        pass

    def eval(self, docs: List[Doc], **kwargs) -> EDMetrics:
        """ 
        A method used for evaluation 
        """
        docs = self.__call__(docs, **kwargs)
        metrics = EDMetrics(docs)
        metrics.summary()
        return metrics

    def train(
        self, 
        train_docs: List[Doc] | None = None, 
        val_docs: List[Doc] | None = None, 
    ):
        """ 
        A method used for training 
        """
        pass