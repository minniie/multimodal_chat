import re
from argparse import ArgumentError

import torch
import numpy as np
from transformers import Trainer

from util.metric import Perplexity, BLEU, DistinctN


class ImageRetrieverTrainer():
    
    def __init__(
            self,
            training_args,
            image_retriever,
            dataset,
            collator
        ):
        self.trainer = Trainer(
            args=training_args,
            model=image_retriever.model,
            train_dataset=dataset["train_set"],
            eval_dataset=dataset["dev_set"],
            data_collator=collator,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
        )
        self.processor = image_retriever.processor

    def run(
            self,
            task_args
        ):
        if task_args.task == "training":
            self.trainer.train()
            self.trainer.save_model()
        elif task_args.task == "evaluation":
            self.trainer.evaluate()
        else:
            raise ArgumentError(f"Task name should be training or evaluation: {task_args.task}")

    @staticmethod
    def preprocess_logits_for_metrics(
            logits,
            labels
        ):
        return logits.to("cpu"), labels.to("cpu")


class ResponseGeneratorTrainer():
    
    def __init__(
            self,
            training_args,
            response_generator,
            dataset,
            collator
        ):
        self.trainer = Trainer(
            args=training_args,
            model=response_generator.model,
            tokenizer=response_generator.tokenizer,
            train_dataset=dataset["train_set"],
            eval_dataset=dataset["dev_set"],
            data_collator=collator,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=self.compute_metrics
        )
        self.tokenizer = response_generator.tokenizer

    def run(
            self,
            task_args
        ):
        if task_args.task == "training":
            self.trainer.train()
            self.trainer.save_model()
        elif task_args.task == "evaluation":
            self.trainer.evaluate()
        else:
            raise ArgumentError(f"Task name should be training or evaluation: {task_args.task}")

    def preprocess_logits_for_metrics(
            self,
            logits,
            labels
        ):
        preds = torch.argmax(logits, axis=-1)
        ppl = Perplexity(logits, labels)["ppl"]

        return ppl, preds.to("cpu"), labels.to("cpu")

    def normalize_decode_per_token(
            self,
            batch
        ):
        batch = batch[:batch.tolist().index(self.tokenizer.eos_token_id)] \
            if self.tokenizer.eos_token_id in batch else batch
        pred = self.tokenizer.batch_decode(np.expand_dims(batch, axis=-1), skip_special_tokens=True)
        pred = [p.lower() for p in pred]
        special_chars = re.compile('[@_!#$%^&*()<>?/\|}{~:.,]')
        pred = list(filter(lambda token: token and not bool(special_chars.match(token)), pred))
        pred = ['NONE'] if not pred else pred

        return pred

    def compute_metrics(
            self,
            prediction
        ):
        ppl, preds, labels = prediction.predictions
        preds[preds == -100] = self.tokenizer.pad_token_id
        labels[labels == -100] = self.tokenizer.pad_token_id
        
        preds_text, labels_text = [], []
        for pred, label in zip(preds, labels):
            preds_text.append(self.normalize_decode_per_token(pred))
            labels_text.append([self.normalize_decode_per_token(label)])
        
        ppl = {"ppl": np.mean(ppl)}
        bleu = BLEU(preds_text, labels_text)
        distinct_n = DistinctN(preds_text, labels_text)
        
        return {**ppl, **bleu, **distinct_n}