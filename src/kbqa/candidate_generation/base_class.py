from typing import Tuple, List, Any
from abc import ABC, abstractmethod
from kbqa.utils.data_types import Doc
from kbqa.utils.metrics import CGMetrics


class EntityCandidateGenerationModel(ABC):
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

    def eval(self, docs: List[Doc], k: int = 30, **kwargs) -> CGMetrics:
        """ 
        A method used for evaluation 
        """
        docs = self.__call__(docs, **kwargs)
        metrics = CGMetrics(docs)
        metrics.summary(k)
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