from typing import List
from abc import ABC, abstractmethod


class AnswerGenerationModel(ABC):
    @abstractmethod
    def generate_answer(self, prompts: List[str]) -> List[str]:
        """
        INPUTS:
            prompts: List[str] - a list of prompts
        OUTPUTS:
            List[str] - a list of answers
        """
        pass