from transformers import Trainer

from learning.callback import ResponseGeneratorCallback


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
            # preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
        )
        self.processor = image_retriever.processor

    def run(
            self,
        ):
        self.trainer.train()
        self.trainer.save_model()

    # @staticmethod
    # def preprocess_logits_for_metrics(
    #         logits,
    #         labels
    #     ):
    #     return logits.to("cpu"), labels.to("cpu")


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
            callbacks=[ResponseGeneratorCallback]
        )

    def run(
            self,
        ):
        self.trainer.train()
        self.trainer.save_model()