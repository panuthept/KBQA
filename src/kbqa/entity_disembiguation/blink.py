import os
import time
import torch
import random
import logging
import numpy as np
from torch import Tensor
from copy import deepcopy
from tqdm import tqdm, trange
from dataclasses import dataclass
from kbqa.utils.metrics import EDMetrics
from typing import Tuple, List, Dict, Any
from kbqa.utils.data_types import Doc, Entity
from kbqa.utils.blink_utils.candidate_ranking import utils
from kbqa.utils.blink_utils.biencoder.biencoder import BiEncoderRanker
from kbqa.entity_disembiguation.base_class import EntityDisambiguationModel
from kbqa.utils.blink_utils.crossencoder.crossencoder import CrossEncoderRanker
from kbqa.utils.blink_utils.crossencoder.data_process import prepare_crossencoder_data
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from kbqa.utils.blink_utils.crossencoder.train_cross import modify, get_optimizer, get_scheduler


@dataclass
class BlinkCrossEncoderConfig:
    # Model configuration
    roberta: str | None = None          # Must provide either roberta or bert_model
    bert_model: str | None = None       # Must provide either roberta or bert_model
    path_to_model: str | None = None    # Path to fine-tuned model
    add_linear: bool = True             # Add linear layer to compute simialrity score between context and candidate
    out_dim: int = 1                    # Score is a scalar value
    pull_from_layer: int = -1           # Pull from the last layer
    lowercase: bool = True
    # GPU configuration
    no_cuda: bool = False
    # Multiple GPU configuration
    data_parallel: bool = False
    # Data configuration
    max_seq_length: int = 160           # Maximum total input length (context + candidate)
    max_cand_length: int = 128          # Maximum candidate length
    max_context_length: int = 32        # Maximum context length
    max_cand_num: int = 30              # Maximum number of candidates
    # Training configuration
    type_optimization: str = "all_encoder_layers"
    learning_rate: float = 3e-5
    fp16: bool = False
    seed: int = 52313
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 5
    warmup_proportion: float = 0.1
    print_interval: int = 10
    eval_interval: int = 2000
    max_grad_norm: float = 1.0
    # Inference configuration
    confident_threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BlinkCrossEncoderConfig":
        # Filter out the keys that are not in the class
        config = {key: value for key, value in config.items() if key in cls.__dataclass_fields__}
        return cls(**config)


class BlinkCrossEncoder(EntityDisambiguationModel):
    def __init__(
            self, 
            entity_corpus: Dict[str, Entity], 
            config: BlinkCrossEncoderConfig,
            entity_pad_id: str = "Q0",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.config = config
        self.crossencoder = CrossEncoderRanker(config.to_dict())
        self.crossencoder.model.to(self.device)
        self.tokenizer = self.crossencoder.tokenizer
        self.id2title = {entity.id: entity.name for entity in entity_corpus.values()}
        self.id2text = {entity.id: entity.desc for entity in entity_corpus.values()}
        self.entity_pad_id = entity_pad_id

    def _preprocess_docs(self, docs: List[Doc], is_training: bool = False, verbose: bool = False):
        max_cand_num = 0
        samples = []
        labels = []
        nns = []
        sample2doc_index = []
        for i, doc in enumerate(tqdm(docs, desc="Preprocessing docs", disable=not verbose)):
            for j, span in enumerate(doc.spans):
                mention = doc.text[span.start:span.start + span.length]
                context_left = doc.text[:span.start]
                context_right = doc.text[span.start + span.length:]
                samples.append({
                    "mention": mention,
                    "context_left": context_left,
                    "context_right": context_right
                })

                in_kb_cand_ids = [entity.id if entity.id in self.id2title and entity.id in self.id2text else self.entity_pad_id for entity in span.cand_entities]
                if is_training and span.gold_entity.id not in in_kb_cand_ids:
                    # NOTE: Skip the sample if the gold entity is not in the candidate list
                    continue

                sample2doc_index.append((i, j))
                labels.append(span.gold_entity.id)
                nns.append(in_kb_cand_ids)
                max_cand_num = max(max_cand_num, len(in_kb_cand_ids))

        if is_training:
            max_cand_num = self.config.max_cand_num

        # Padding the candidates
        for nn in nns:
            nn.extend([self.entity_pad_id] * (max_cand_num - len(nn)))
        padding_masks = torch.tensor([[entity_id != self.entity_pad_id for entity_id in nn] for nn in nns])
        
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            self.tokenizer, 
            samples, 
            labels, 
            nns, 
            self.id2title, 
            self.id2text, 
            keep_all=not is_training,   # NOTE: If this is False, the implementation of sample2doc_index can be wrong. Thus do not evaluate on training data.
            verbose=verbose,
        )
        context_input = modify(
            context_input, candidate_input, self.config.max_seq_length
        )
        tensor_data = TensorDataset(context_input, label_input, padding_masks)
        return tensor_data, sample2doc_index
    
    def _process_inputs(
            self, 
            docs: List[Doc],
            batch_size: int = 1,
            is_training: bool = False,
            verbose: bool = False,
    ) -> Tuple[DataLoader, List[Tuple[int, int]]]:
        tensor_data, sample2doc_index = self._preprocess_docs(docs, is_training=is_training, verbose=verbose)
        sampler = SequentialSampler(tensor_data) if not is_training else RandomSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=sampler, batch_size=batch_size
        )
        return dataloader, sample2doc_index
    
    def forward(
            self, 
            batch, 
            is_training: bool = False, 
    ) -> Tuple[Tensor, Tensor]:
        batch = tuple(t.to(self.device) for t in batch)
        context_input, label_input, padding_masks = batch
        if is_training:
            loss, logits = self.crossencoder(context_input, label_input, self.config.max_context_length)
        else:
            with torch.no_grad():
                loss, logits = self.crossencoder(context_input, label_input, self.config.max_context_length)
        logits = torch.where(padding_masks, logits, torch.tensor(-1e9).to(self.device))
        return loss, logits
    
    def __call__(
            self, 
            docs: List[Doc],
            dataloader: DataLoader | None = None,                   # NOTE: If dataloader is provided, sample2doc_index must be provided
            sample2doc_index: List[Tuple[int, int]] | None = None,  # NOTE: If dataloader is provided, sample2doc_index must be provided
            batch_size: int = 1,
            verbose: bool = False,
    ) -> List[Doc]:
        self.crossencoder.model.eval()

        if dataloader is None and sample2doc_index is None:
            dataloader, sample2doc_index = self._process_inputs(docs, batch_size=batch_size, is_training=False, verbose=verbose)

        all_pred_scores = []
        all_pred_indices = []
        all_cand_scores = []
        for batch in tqdm(dataloader, disable=not verbose, desc="Inference"):
            _, logits = self.forward(batch, is_training=False)
            norm_logits = torch.nn.functional.softmax(logits, dim=-1)
            pred_scores, pred_indices = torch.max(norm_logits, dim=-1)
            all_cand_scores.extend(norm_logits.cpu().numpy().tolist())
            all_pred_scores.extend(pred_scores.cpu().numpy().tolist())
            all_pred_indices.extend(pred_indices.cpu().numpy().tolist())
        
        pred_docs = deepcopy(docs)
        for (doc_idx, span_idx), pred_score, pred_idx, cand_scores in zip(sample2doc_index, all_pred_scores, all_pred_indices, all_cand_scores):
            pred_entity = Entity(id=pred_docs[doc_idx].spans[span_idx].cand_entities[pred_idx].id, score=pred_score) if pred_score >= self.config.confident_threshold else Entity(id=self.entity_pad_id, score=pred_score)
            pred_docs[doc_idx].spans[span_idx].pred_entity = pred_entity
            for cand_idx in range(len(pred_docs[doc_idx].spans[span_idx].cand_entities)):
                pred_docs[doc_idx].spans[span_idx].cand_entities[cand_idx].score = cand_scores[cand_idx]
        return pred_docs
    
    def save_checkpoint(
            self, 
            checkpoint_path: str, 
            optimizer, 
            scheduler, 
            epoch: int = 0, 
            val_best_score: float = 0.0,
            best_model_path: str = None,
    ):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint = {
            "epoch": epoch,
            "val_best_score": val_best_score,
            "best_model_path": best_model_path,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(checkpoint_path, "checkpoint.pt"))
        self.crossencoder.save(checkpoint_path)

    def load_checkpoint(
            self, 
            checkpoint_path: str, 
            optimizer, 
            scheduler
    ):
        checkpoint = torch.load(os.path.join(checkpoint_path, "checkpoint.pt"))
        epoch = checkpoint["epoch"]
        val_best_score = checkpoint["val_best_score"]
        best_model_path = checkpoint["best_model_path"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        self.crossencoder.load_model(checkpoint_path)
        return epoch, val_best_score, best_model_path, optimizer, scheduler
        
    def train(
            self, 
            train_docs: List[Doc] | None = None, 
            train_dataloader: DataLoader | None = None,
            val_docs: List[Doc] | None = None, 
            batch_size: int = 1,
            resume: bool = False,
            checkpoint_path: str = None,
            model_output_path: str = "models/blink_crossencoder",
    ):
        assert train_docs is not None or train_dataloader is not None, "Either train_docs or train_dataloader must be provided."

        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)

        logging.basicConfig(filename=os.path.join(model_output_path, "train.log"), filemode="w", level=logging.INFO)
        logger = logging.getLogger("BlinkCrossEncoder")

        # Fix the random seeds
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.crossencoder.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

        # Prepare training data
        if train_dataloader is None:
            train_dataloader, _ = self._process_inputs(train_docs, batch_size=batch_size, is_training=True)

        # Prepare optimizer and sheduler
        optimizer = get_optimizer(self.crossencoder.model, self.config.to_dict())
        scheduler = get_scheduler(self.config.to_dict(), optimizer, len(train_docs), logger)

        resume_epoch = None
        val_best_score = 0.0
        best_model_path = None
        if resume and checkpoint_path:
            resume_epoch, val_best_score, best_model_path, optimizer, scheduler = self.load_checkpoint(checkpoint_path, optimizer, scheduler)
            logger.info(f" Resuming training from epoch {resume_epoch + 1} ...")
        
        # Prepare validation data
        if val_docs:
            val_dataloader, val_sample2doc_index = self._process_inputs(val_docs, batch_size=batch_size, is_training=False)
            if not (resume and checkpoint_path):
                logger.info(" Evaluating on validation data (initial) ...")
                pred_val_docs = self.__call__(val_docs, val_dataloader, val_sample2doc_index, batch_size=batch_size)
                metrics = EDMetrics(pred_val_docs)
                metrics.summary(logger)
                val_best_score = metrics.get_f1()
                save_model_path = os.path.join(model_output_path, f"epoch_0_{round(val_best_score * 100, 1)}")
                logger.info(f" Saving initial model to {save_model_path}")
                utils.save_model(self.crossencoder.model, self.tokenizer, save_model_path)
                best_model_path = save_model_path

        # Training loop
        logger.info(" Starting training...")
        logger.info(f" device: {self.device}")
        utils.write_to_file(
            os.path.join(model_output_path, "training_params.txt"), str(self.config.to_dict())
        )
        time_start = time.time()

        self.crossencoder.model.train()

        num_train_epochs = self.config.num_train_epochs
        grad_acc_steps = self.config.gradient_accumulation_steps
        for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
            if resume_epoch is not None and epoch_idx < resume_epoch:
                continue
            part = 0
            train_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Batch")):
                loss, _ = self.forward(batch, is_training=True)

                if grad_acc_steps > 1:
                    loss = loss / grad_acc_steps
                
                train_loss += loss.item()
                loss.backward()

                if (step + 1) % (self.config.print_interval * grad_acc_steps) == 0:
                    train_loss = train_loss / (self.config.print_interval * grad_acc_steps)
                    logger.info(f" Epoch: {epoch_idx + 1}, Step: {step + 1}, Train Loss: {train_loss}")
                    train_loss = 0

                if (step + 1) % grad_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.crossencoder.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (step + 1) % (self.config.eval_interval * grad_acc_steps) == 0:
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

            # Save checkpoint
            checkpoint_path = os.path.join(model_output_path, f"checkpoint_{epoch_idx}")
            self.save_checkpoint(checkpoint_path, optimizer, scheduler, epoch_idx, val_best_score, best_model_path)

        execution_time = (time.time() - time_start) / 60
        utils.write_to_file(
            os.path.join(model_output_path, "training_time.txt"), f"Execution time: {execution_time} minutes"
        )
        logger.info(f" Training completed in {execution_time} minutes")
        logger.info(f" Best model path: {best_model_path}")
        self.config.path_to_model = best_model_path


@dataclass
class BlinkCrossEncoderCFTConfig(BlinkCrossEncoderConfig):
    cft_weight: float = 0.1


class BlinkCrossEncoderCFT(BlinkCrossEncoder):
    def mask_context_input(self, context_input: Tensor):
        context_input = context_input.clone()

        start_indices = (context_input == 1).nonzero(as_tuple=True)[-1]
        end_indices = (context_input == 2).nonzero(as_tuple=True)[-1]
        for sample_idx, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
            context_input[sample_idx, start_idx + 1:end_idx] = self.tokenizer.mask_token_id
        return context_input

    def _preprocess_docs(self, docs: List[Doc], is_training: bool = False, verbose: bool = False):
        max_cand_len = 0
        samples = []
        labels = []
        nns = []
        sample2doc_index = []
        for i, doc in enumerate(tqdm(docs, desc="Preprocessing docs", disable=not verbose)):
            for j, span in enumerate(doc.spans):
                mention = doc.text[span.start:span.start + span.length]
                context_left = doc.text[:span.start]
                context_right = doc.text[span.start + span.length:]
                samples.append({
                    "mention": mention,
                    "context_left": context_left,
                    "context_right": context_right
                })
                in_kb_cand_ids = [entity.id if entity.id in self.id2title and entity.id in self.id2text else self.entity_pad_id for entity in span.cand_entities]
                sample2doc_index.append((i, j))
                labels.append(span.gold_entity.id)
                nns.append(in_kb_cand_ids)
                max_cand_len = max(max_cand_len, len(in_kb_cand_ids))
        
        # Padding the candidates
        for nn in nns:
            nn.extend([self.entity_pad_id] * (max_cand_len - len(nn)))
        padding_masks = torch.tensor([[entity_id != self.entity_pad_id for entity_id in nn] for nn in nns])
        
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            self.tokenizer, 
            samples, 
            labels, 
            nns, 
            self.id2title, 
            self.id2text, 
            keep_all=not is_training   # NOTE: If this is False, the implementation of sample2doc_index can be wrong. Thus do not evaluate on training data.
        )
        cft_context_input = self.mask_context_input(context_input) if is_training else context_input

        context_input = modify(
            context_input, candidate_input, self.config.max_seq_length
        )
        cft_context_input = modify(
            cft_context_input, candidate_input, self.config.max_seq_length
        ) if is_training else context_input

        tensor_data = TensorDataset(context_input, cft_context_input, label_input, padding_masks)
        return tensor_data, sample2doc_index
    
    def forward(
            self, 
            batch, 
            is_training: bool = False, 
    ) -> Tuple[Tensor, Tensor]:
        batch = tuple(t.to(self.device) for t in batch)
        context_input, cft_context_input, label_input, padding_masks = batch
        if is_training:
            loss, logits = self.crossencoder(context_input, label_input, self.config.max_context_length)
            cft_loss, _ = self.crossencoder(cft_context_input, label_input, self.config.max_context_length)
            loss = loss + self.config.cft_weight * cft_loss
        else:
            with torch.no_grad():
                loss, logits = self.crossencoder(context_input, label_input, self.config.max_context_length)
        logits = torch.where(padding_masks, logits, torch.tensor(-1e9).to(self.device))
        return loss, logits


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
    from kbqa.utils.data_utils import get_entity_corpus

    entity_corpus = get_entity_corpus("./data/entity_corpus.jsonl")
    config = BlinkCrossEncoderConfig.from_dict(json.load(open("crossencoder_wiki_large.json")))
    config.path_to_model = "./crossencoder_wiki_large.bin"

    # config = json.load(open("crossencoder_wiki_large.json"))
    # config["cft_weight"] = 0.1
    # blink_crossencoder = BlinkCrossEncoderCFT(entity_corpus, config)
    blink_crossencoder = BlinkCrossEncoder(entity_corpus, config)

    train_docs = [
        Doc(
            text="Is Joe Biden the president of the United States?",
            spans=[
                Span(
                    start=3, 
                    length=9, 
                    surface_form="Joe Biden",
                    gold_entity=Entity(id="Q6279"),
                    cand_entities=[
                        Entity(id="Q6279"),
                        Entity(id="Q65053339"),
                        Entity(id="Q63241885"),
                    ]
                ),
                Span(
                    start=34, 
                    length=13, 
                    surface_form="United States",
                    gold_entity=Entity(id="Q30"),
                    cand_entities=[
                        Entity(id="Q30"),
                        Entity(id="Q11268"),
                        Entity(id="Q35657"),
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
                    gold_entity=Entity(id="Q142"),
                    cand_entities=[
                        Entity(id="Q47774"),
                        Entity(id="Q142"),
                        Entity(id="Q193563"),
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
                    gold_entity=Entity(id="Q3308285"),
                    cand_entities=[
                        Entity(id="Q41421"),
                        Entity(id="Q27069141"),
                        Entity(id="Q1928047"),
                        Entity(id="Q65029442"),
                        Entity(id="Q108883102"),
                        Entity(id="Q3308285"),
                    ]
                ),
                Span(
                    start=40, 
                    length=16, 
                    surface_form="machine learning",
                    gold_entity=Entity(id="Q2539"),
                    cand_entities=[
                        Entity(id="Q2539"),
                        Entity(id="Q6723676"),
                        Entity(id="Q108371168"),
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
                    gold_entity=Entity(id="Q41421"),
                    cand_entities=[
                        Entity(id="Q41421"),
                        Entity(id="Q27069141"),
                        Entity(id="Q1928047"),
                        Entity(id="Q65029442"),
                        Entity(id="Q108883102"),
                        Entity(id="Q3308285"),
                    ]
                )
            ]
        )
    ]

    pred_docs = blink_crossencoder(train_docs)
    print(pred_docs)
    # blink_crossencoder.eval(test_docs)
    # blink_crossencoder.train(train_docs, val_docs=test_docs, batch_size=10)