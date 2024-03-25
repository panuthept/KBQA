from typing import List
from abc import ABC, abstractmethod


class KnowledgeRetrievalModel(ABC):
    @abstractmethod
    def retrieve_knowledge(self, queries: List[str]) -> List[str]:
        """
        INPUTS:
            queries: List[str] - a list of queries
        OUTPUTS:
            List[str] - a list of retrieved knowledges
        """
        pass