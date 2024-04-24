import os
import time
import torch
import random
import logging
import numpy as np
from torch import Tensor
from copy import deepcopy
from tqdm import tqdm, trange
from kbqa.utils.metrics import EDMetrics
from kbqa.utils.data_types import Doc, Entity
from typing import Tuple, Iterator, List, Dict, Any
from kbqa.ed.models.blink_utils.candidate_ranking import utils
from kbqa.ed.models.base_class import EntityDisambiguationModel
from kbqa.ed.models.blink_utils.biencoder.biencoder import BiEncoderRanker
from kbqa.ed.models.blink_utils.crossencoder.crossencoder import CrossEncoderRanker
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from kbqa.ed.models.blink_utils.crossencoder.data_process import prepare_crossencoder_data
from kbqa.ed.models.blink_utils.crossencoder.train_cross import modify, get_optimizer, get_scheduler


class BlinkCrossEncoder(EntityDisambiguationModel):
    def __init__(
            self, 
            entity_corpus: Dict[str, Entity], 
            config: Dict[str, Any], 
            checkpoint_path: str = None,
            confident_threshold: float = 0.0,
            entity_pad_id: str = "Q0",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.config = config
        self.config["confident_threshold"] = confident_threshold
        self.config["path_to_model"] = checkpoint_path
        self.crossencoder = CrossEncoderRanker(config)
        self.crossencoder.model.to(self.device)
        self.tokenizer = self.crossencoder.tokenizer
        self.id2title = {entity_id: entity.name for entity_id, entity in entity_corpus.items()}
        self.id2text = {entity_id: entity.desc for entity_id, entity in entity_corpus.items()}
        self.entity_pad_id = entity_pad_id

    @staticmethod
    def _preprocess_docs(docs: List[Doc]):
        samples = []
        labels = []
        nns = []
        sample2doc_index = []
        for i, doc in enumerate(docs):
            for j, span in enumerate(doc.spans):
                mention = doc.text[span.start:span.start + span.length]
                context_left = doc.text[:span.start]
                context_right = doc.text[span.start + span.length:]
                samples.append({
                    "mention": mention,
                    "context_left": context_left,
                    "context_right": context_right
                })
                labels.append(span.gold_entity.id)
                nns.append([entity.id for entity in span.cand_entities])
                sample2doc_index.append((i, j))
        return samples, labels, nns, sample2doc_index

    @staticmethod
    def _process_crossencoder_dataloader(
            context_input, 
            label_input, 
            batch_size: int = 1,
            sampler_method: str = "sequential",
    ):
        assert sampler_method in ["sequential", "random"], "Invalid sampler method, must be either 'sequential' or 'random'"
        tensor_data = TensorDataset(context_input, label_input)
        sampler = SequentialSampler(tensor_data) if sampler_method == "sequential" else RandomSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=sampler, batch_size=batch_size
        )
        return dataloader
    
    def _process_inputs(
            self, 
            docs: List[Doc] | Iterator[Doc],
            batch_size: int = 1,
            is_training: bool = False,
    ) -> Tuple[DataLoader, List[Tuple[int, int]]]:
        samples, labels, nns, sample2doc_index = self._preprocess_docs(docs)
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            self.tokenizer, 
            samples, 
            labels, 
            nns, 
            self.id2title, 
            self.id2text, 
            keep_all=not is_training   # NOTE: If this is False, the implementation of sample2doc_index can be wrong. Thus do not evaluate on training data.
        )
        context_input = modify(
            context_input, candidate_input, self.config["max_seq_length"]
        )
        dataloader = self._process_crossencoder_dataloader(
            context_input, label_input, batch_size=batch_size, sampler_method="random" if is_training else "sequential"
        )
        return dataloader, sample2doc_index
    
    def forward(
            self, 
            batch, 
            is_training: bool = False, 
    ) -> Tuple[Tensor, Tensor]:
        batch = tuple(t.to(self.device) for t in batch)
        context_input, label_input = batch
        if is_training:
            loss, logits = self.crossencoder(context_input, label_input, self.config["max_context_length"])
        else:
            with torch.no_grad():
                loss, logits = self.crossencoder(context_input, label_input, self.config["max_context_length"])
        return loss, logits
    
    def __call__(
            self, 
            docs: List[Doc] | Iterator[Doc],
            dataloader: DataLoader | None = None,
            sample2doc_index: List[Tuple[int, int]] | None = None,
            batch_size: int = 1,
    ) -> List[Doc]:
        self.crossencoder.model.eval()

        # Convert docs to list of it is an Iterator
        if isinstance(docs, Iterator):
            output_docs = deepcopy([doc for doc in docs])
        else:
            output_docs = deepcopy(docs)

        if dataloader is None and sample2doc_index is None:
            dataloader, sample2doc_index = self._process_inputs(docs, batch_size=batch_size, is_training=False)

        all_pred_scores = []
        all_pred_indices = []
        for batch in dataloader:
            _, logits = self.forward(batch, is_training=False)
            norm_logits = torch.nn.functional.softmax(logits, dim=-1)
            pred_scores, pred_indices = torch.max(norm_logits, dim=-1)
            all_pred_scores.extend(pred_scores.cpu().numpy().tolist())
            all_pred_indices.extend(pred_indices.cpu().numpy().tolist())
        
        for (doc_idx, span_idx), pred_score, pred_idx in zip(sample2doc_index, all_pred_scores, all_pred_indices):
            output_docs[doc_idx].spans[span_idx].pred_entity = output_docs[doc_idx].spans[span_idx].cand_entities[pred_idx] if pred_score >= self.config["confident_threshold"] else self.entity_pad_id
            output_docs[doc_idx].spans[span_idx].pred_score = pred_score
        return output_docs
    
    def train(
            self, 
            train_docs: List[Doc] | Iterator[Doc], 
            val_docs: List[Doc] | None = None, 
            batch_size: int = 1,
            model_output_path: str = "models/blink_crossencoder",
    ):
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)

        logging.basicConfig(filename=os.path.join(model_output_path, "train.log"), filemode="w", level=logging.INFO)
        logger = logging.getLogger("BlinkCrossEncoder")

        # Fix the random seeds
        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.crossencoder.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

        # Prepare training data
        train_dataloader, _ = self._process_inputs(train_docs, batch_size=batch_size, is_training=True)

        # Prepare validation data
        best_model_path = None
        val_best_score = None
        if val_docs:
            val_dataloader, val_sample2doc_index = self._process_inputs(val_docs, batch_size=batch_size, is_training=False)
            logger.info(" Evaluating on validation data (initial) ...")
            pred_val_docs = self.__call__(val_docs, val_dataloader, val_sample2doc_index, batch_size=batch_size)
            metrics = EDMetrics(pred_val_docs)
            metrics.summary(logger)
            val_best_score = metrics.get_f1()
            save_model_path = os.path.join(model_output_path, f"epoch_0_{round(val_best_score * 100, 1)}")
            logger.info(f" Saving initial model to {save_model_path}")
            utils.save_model(self.crossencoder.model, self.tokenizer, save_model_path)
            best_model_path = save_model_path

        # Prepare optimizer and sheduler
        optimizer = get_optimizer(self.crossencoder.model, self.config)
        scheduler = get_scheduler(self.config, optimizer, len(train_docs), logger)

        # Training loop
        logger.info(" Starting training...")
        logger.info(f" device: {self.device}")
        utils.write_to_file(
            os.path.join(model_output_path, "training_params.txt"), str(self.config)
        )
        time_start = time.time()

        self.crossencoder.model.train()

        num_train_epochs = self.config["num_train_epochs"]
        grad_acc_steps = self.config["gradient_accumulation_steps"]
        for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
            part = 0
            train_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Batch")):
                loss, _ = self.forward(batch, is_training=True)

                if grad_acc_steps > 1:
                    loss = loss / grad_acc_steps
                
                train_loss += loss.item()
                loss.backward()

                if (step + 1) % (self.config["print_interval"] * grad_acc_steps) == 0:
                    train_loss = train_loss / (self.config["print_interval"] * grad_acc_steps)
                    logger.info(f" Epoch: {epoch_idx + 1}, Step: {step + 1}, Train Loss: {train_loss}")
                    train_loss = 0

                if (step + 1) % grad_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.crossencoder.model.parameters(), self.config["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (step + 1) % (self.config["eval_interval"] * grad_acc_steps) == 0:
                    if val_docs:
                        logger.info(f" Evaluating on validation data (epoch_{epoch_idx + 1}_{part + 1}) ...")
                        pred_val_docs = self.__call__(val_docs, val_dataloader, val_sample2doc_index, batch_size=batch_size)
                        metrics = EDMetrics(pred_val_docs)
                        metrics.summary(logger)
                        val_score = metrics.get_f1()
                        if val_score > val_best_score:
                            val_best_score = val_score
                            save_model_path = os.path.join(model_output_path, f"epoch_{epoch_idx + 1}_{part + 1}_f1_{round(val_best_score * 100, 1)}")
                            logger.info(f" Saving fine-tuned model to {save_model_path}")
                            utils.save_model(self.crossencoder.model, self.tokenizer, save_model_path)
                            best_model_path = save_model_path
                        self.crossencoder.model.train()
                    else:
                        save_model_path = os.path.join(model_output_path, f"epoch_{epoch_idx + 1}_{part + 1}")
                        logger.info(f" Saving fine-tuned model to {save_model_path}")
                        utils.save_model(self.crossencoder.model, self.tokenizer, save_model_path)
                        best_model_path = save_model_path
                    part += 1
            
            if val_docs:
                logger.info(f" Evaluating on validation data (epoch_{epoch_idx + 1}) ...")
                pred_val_docs = self.__call__(val_docs, val_dataloader, val_sample2doc_index, batch_size=batch_size)
                metrics = EDMetrics(pred_val_docs)
                metrics.summary(logger)
                val_score = metrics.get_f1()
                if val_score > val_best_score:
                    val_best_score = val_score
                    save_model_path = os.path.join(model_output_path, f"epoch_{epoch_idx + 1}_f1_{round(val_best_score * 100, 1)}")
                    logger.info(f" Saving fine-tuned model to {save_model_path}")
                    utils.save_model(self.crossencoder.model, self.tokenizer, save_model_path)
                    best_model_path = save_model_path
                self.crossencoder.model.train()
            else:
                save_model_path = os.path.join(model_output_path, f"epoch_{epoch_idx + 1}")
                logger.info(f" Saving fine-tuned model to {save_model_path}")
                utils.save_model(self.crossencoder.model, self.tokenizer, save_model_path)
                best_model_path = save_model_path

        execution_time = (time.time() - time_start) / 60
        utils.write_to_file(
            os.path.join(model_output_path, "training_time.txt"), f"Execution time: {execution_time} minutes"
        )
        logger.info(f" Training completed in {execution_time} minutes")
        logger.info(f" Best model path: {best_model_path}")
        self.config["path_to_model"] = best_model_path


# class BlinkBiencoder(EntityDisambiguationModel):
#     def __init__(
#             self, 
#             entity_corpus: Dict[str, Any], 
#             config: Dict[str, Any], 
#             checkpoint_path: str=None
#     ):
#         config["path_to_model"] = checkpoint_path
#         self.biencoder = BiEncoderRanker(config)
#         self.tokenizer = self.biencoder.tokenizer
#         self.entity_corpus = entity_corpus

#     def __call__(self, docs: List[Doc]):
#         samples = preprocess_docs(docs)


if __name__ == "__main__":
    import json
    from kbqa.utils.data_types import Span

    entity_corpus = {
        "0": Entity(id="0", name="<PAD>", desc="<PAD>"),
        "1": Entity(id="1", name="Joe Biden", desc="46th president of the United States"),
        "2": Entity(id="2", name="Joe Biden", desc="fictional parody character from The Onion loosely based on the real Joe Biden"),
        "3": Entity(id="3", name="United States", desc="country primarily located in North America"),
        "4": Entity(id="4", name="United States dollar", desc="currency of the United States"),
        "5": Entity(id="5", name="United States Army", desc="land warfare service branch of the United States Armed Forces"),
        "6": Entity(id="6", name="France", desc="country primarily located in Western Europe"),
        "7": Entity(id="7", name="France national football team", desc="soccer team representing France"),
        "8": Entity(id="8", name="Michael Jordan", desc="American former professional basketball player"),
        "9": Entity(id="9", name="Michael Jordan", desc="American computer scientist, University of California, Berkeley"),
        "10": Entity(id="10", name="Michael Jordan", desc="German draughtsperson, artist and comics artist"),
        "11": Entity(id="11", name="machine learning", desc="scientific study of algorithms and statistical models that computer systems use to perform tasks without explicit instructions"),
        "12": Entity(id="12", name="machine learning", desc="journal"),
    }

    config = json.load(open("crossencoder_wiki_large.json"))
    # blink_crossencoder = BlinkCrossEncoder(entity_corpus, config, checkpoint_path="crossencoder_wiki_large.bin")
    blink_crossencoder = BlinkCrossEncoder(entity_corpus, config)

    train_docs = [
        Doc(
            text="Is Joe Biden the president of the United States?",
            spans=[
                Span(
                    start=3, 
                    length=9, 
                    surface_form="Joe Biden",
                    gold_entity=Entity(id="1"),
                    cand_entities=[
                        Entity(id="1"),
                        Entity(id="2"),
                        Entity(id="0"),
                    ]
                ),
                Span(
                    start=34, 
                    length=13, 
                    surface_form="United States",
                    gold_entity=Entity(id="3"),
                    cand_entities=[
                        Entity(id="3"),
                        Entity(id="4"),
                        Entity(id="5"),
                    ]
                )
            ]
        ),
        Doc(
            text="What is the capital of France?",
            spans=[
                Span(
                    start=23, 
                    length=6, 
                    surface_form="France",
                    gold_entity=Entity(id="6"),
                    cand_entities=[
                        Entity(id="6"),
                        Entity(id="7"),
                        Entity(id="0"),
                    ]
                )
            ]
        ),
        Doc(
            text="Michael Jordan published a new paper on machine learning.",
            spans=[
                Span(
                    start=0, 
                    length=14, 
                    surface_form="Michael Jordan",
                    gold_entity=Entity(id="9"),
                    cand_entities=[
                        Entity(id="8"),
                        Entity(id="9"),
                        Entity(id="10"),
                    ]
                ),
                Span(
                    start=40, 
                    length=16, 
                    surface_form="machine learning",
                    gold_entity=Entity(id="11"),
                    cand_entities=[
                        Entity(id="11"),
                        Entity(id="12"),
                        Entity(id="0"),
                    ]
                )
            ]
        ),
    ]
    test_docs = [
        Doc(
            text="What year did Michael Jordan win his first NBA championship?",
            spans=[
                Span(
                    start=14, 
                    length=14, 
                    surface_form="Michael Jordan",
                    gold_entity=Entity(id="8"),
                    cand_entities=[
                        Entity(id="8"),
                        Entity(id="9"),
                        Entity(id="10"),
                    ]
                )
            ]
        )
    ]

    def get_Iterator(docs: List[Doc]) -> Iterator[Doc]:
        for doc in docs:
            yield doc

    iter_train_docs = get_Iterator(train_docs)
    pred_docs = blink_crossencoder(iter_train_docs)
    print(pred_docs)
    # pred_docs = blink_crossencoder(train_docs)
    # print(pred_docs)
    # blink_crossencoder.eval(test_docs)
    # blink_crossencoder.train(train_docs, val_docs=test_docs, batch_size=10)