import re

import numpy as np
import torch
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
            data_collator=collator
        )
        self.processor = image_retriever.processor

    def train(
            self
        ):
        self.trainer.train()
        self.trainer.save_model()

    def inference(
            self
        ):
        pass


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
            compute_metrics=self.compute_metrics
        )
        self.tokenizer = response_generator.tokenizer

    def train(
            self
        ):
        self.trainer.train()
        self.trainer.save_model()

    def inference(
            self
        ):
        pass

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
        logits, labels = prediction.predictions.copy(), prediction.label_ids.copy()
        preds = np.argmax(logits, axis=-1)
        labels_original = labels.copy()
        preds[preds == -100] = self.tokenizer.pad_token_id
        labels[labels == -100] = self.tokenizer.pad_token_id
        
        preds_text, labels_text = [], []
        for pred, label in zip(preds, labels):
            preds_text.append(self.normalize_decode_per_token(pred))
            labels_text.append([self.normalize_decode_per_token(label)])
        
        ppl = Perplexity(logits, labels_original)
        bleu = BLEU(preds_text, labels_text)
        distinct_n = DistinctN(preds_text, labels_text)
        
        return {**ppl, **bleu, **distinct_n}