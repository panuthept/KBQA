from typing import List, Dict, Any
from kbqa.answer_generation_models.base_class import AnswerGenerationModel
from kbqa.knowledge_retrieval_models.base_class import KnowledgeRetrievalModel


class GeneratorConfig:
    name: str
    checkpoint_path: str


class KBQA:
    def __init__(
            self, 
            answer_generation_name: str = "ChatGPT",
            answer_generation_checkpoint_path: str | None = None,
            answer_generation_model: AnswerGenerationModel | None = None,
            knowledge_retrieval_name: str = "Wikipedia_mE5", 
            knowledge_retriever_checkpoint_path: str | None = None,
            knowledge_retrieval_model: KnowledgeRetrievalModel | None = None
    ):
        """
        INPUTS:
            knowledge_retriever_path_or_name: str | None - a path or name of knowledge retriever
        """
        self.answer_generation_model = self.load_answer_generator(
            name=answer_generation_name, 
            checkpoint_path=answer_generation_checkpoint_path
        ) if answer_generation_model is None else answer_generation_model

        self.knowledge_retrieval_model = self.load_knowledge_retriever(
            name=knowledge_retrieval_name,
            checkpoint_path=knowledge_retriever_checkpoint_path
        ) if knowledge_retrieval_model is None else knowledge_retrieval_model

    def __call__(
            self, 
            questions: List[str], 
            return_knowledge: bool = False, 
            return_promt: bool = False
    ) -> Dict[str, Any]:
        """
        INPUTS:
            questions: List[str] - a list of questions
        OUTPUTS:
            Dict[str, Any] - a dictionary of results (Knowledges, Answers)
        """
        response = {}
        knowledges = self.knowledge_retrieval_model.retrieve_knowledge(questions)
        if return_knowledge:
            response["knowledge"] = knowledges
        prompts = self.create_promts(questions, knowledges)
        if return_promt:
            response["prompts"] = prompts
        answers = self.answer_generation_model.generate_answer(prompts)
        response["answer"] = answers
        return response
    
    def create_promts(self, questions: List[str], knowledges: List[str] | None = None) -> List[str]:
        """
        INPUTS:
            knowledges: List[str] - a list of supporting evidences
        OUTPUTS:
            List[str] - a list of prompts
        """
        if knowledges is None:
            knowledges = [None] * len(questions)

        promts = []
        for q, k in zip(questions, knowledges):
            promt = ""
            if k is not None:
                promt += f"Based on the following information:\n\n{k}\n\n"
            promt += f"Answer the following question.\nQuestion:{q}\nAnswer:"
            promts.append(promt)
        return promts

    def load_answer_generator(self, name: str = "ChatGPT", checkpoint_path: str | None = None) -> AnswerGenerationModel:
        """
        INPUTS:
            answer_generator: AnswerGenerationModel - an answer generation model
        """
        pass

    def load_knowledge_retriever(self, name: str = "Wikipedia_mE5", checkpoint_path: str | None = None) -> KnowledgeRetrievalModel:
        """
        INPUTS:
            retriever: KnowledgeRetrievalModel - a knowledge retrieval model
        """
        pass