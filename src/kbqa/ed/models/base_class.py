from typing import Tuple, List, Any
from abc import ABC, abstractmethod
from kbqa.utils.data_types import Doc
from torch.utils.data import DataLoader
from kbqa.utils.metrics import EDMetrics


class EntityDisambiguationModel(ABC):
    @abstractmethod
    def _process_inputs(self, docs: List[Doc]) -> Any:
        """ 
        A method used for processing docs into a format that can be used for self.forward() 
        """
        pass

    @abstractmethod
    def forward(self, batch: Any) -> Tuple[Any, Any]:
        """ 
        A method used for computing predictions and losses
        """
        pass

    @abstractmethod
    def __call__(
        self, 
        docs: List[Doc], 
        dataloader: DataLoader | None = None, 
        batch_size: int = 1
    ) -> List[Doc]:
        """ 
        A method used for inference 
        """

    def eval(self, docs: List[Doc], batch_size: int = 1) -> EDMetrics:
        """ 
        A method used for evaluation 
        """
        docs = self.__call__(docs, batch_size=batch_size)
        metrics = EDMetrics(docs)
        metrics.summary()
        return metrics

    @abstractmethod
    def train(
        self, 
        train_docs: List[Doc] | None = None, 
        train_dataloader: DataLoader | None = None, 
        val_docs: List[Doc] | None = None, 
        batch_size: int = 1
    ):
        """ 
        A method used for training 
        """
        pass