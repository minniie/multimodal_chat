from argparse import ArgumentError

from transformers import Trainer

from learning.callback import MetricCallback


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
            callbacks=[MetricCallback]
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