import re

import numpy as np
import torch
from transformers import Trainer, EvalPrediction
from nltk.translate.bleu_score import corpus_bleu


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

        return pred

    @staticmethod
    def preprocess_logits_for_metrics(
            logits,
            labels
        ):
        preds = torch.argmax(logits, dim=-1)
        return preds, labels

    def compute_metrics(
            self,
            prediction
        ):
        preds, labels = prediction.predictions[0].copy(), prediction.label_ids.copy()
        preds[preds == -100] = self.tokenizer.pad_token_id
        labels[labels == -100] = self.tokenizer.pad_token_id
        
        preds_text, labels_text = [], []
        for pred, label in zip(preds, labels):
            preds_text.append(self.normalize_decode_per_token(pred))
            labels_text.append(self.normalize_decode_per_token(label))
        # print(preds_text)
        # print(labels_text)
        bleu = corpus_bleu(labels_text, preds_text, weights=(1,0,0,0))
        
        # preds_text = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # labels_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # bleu = corpus_bleu(labels_text, preds_text, weights=(1,0,0,0))
        
        return {"bleu": bleu}