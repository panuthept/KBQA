from abc import ABC, abstractmethod
from kbqa.utils.data_types import Doc
from kbqa.utils.metrics import EDMetrics
from typing import Iterator, Dict, List, Any


class EntityDisambiguationModel(ABC):
    @abstractmethod
    def _process_inputs(self, docs: List[Doc] | Iterator[Doc]) -> Any:
        """ 
        A method used for processing docs into a format that can be used for self.forward() 
        """
        pass

    @abstractmethod
    def forward(self, batch: Any, return_loss: bool = False) -> Dict[str, Any]:
        """ 
        A method used for computing predictions and losses (optional)
        """
        pass

    @abstractmethod
    def __call__(self, docs: List[Doc] | Iterator[Doc], batch_size: int = 1) -> List[Doc]:
        """ 
        A method used for inference 
        """

    def eval(self, docs: List[Doc] | Iterator[Doc], batch_size: int = 1):
        """ 
        A method used for evaluation 
        """
        docs = self.__call__(docs, batch_size=batch_size)
        metrics = EDMetrics(docs)
        metrics.summary()

    @abstractmethod
    def train(self, train_docs: List[Doc] | Iterator[Doc], val_docs: List[Doc] | Iterator[Doc] | None = None, batch_size: int = 1):
        """ 
        A method used for training 
        """
        pass