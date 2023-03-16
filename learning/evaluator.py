import torch
from transformers import Trainer

from learning.callback import ResponseGeneratorCallback
from util.image import load_image_from_url
from util.text import join_dialog


class ImageRetrieverEvaluator():
    
    def __init__(
            self,
            device,
            data_args,
            image_retriever,
            dataset
        ):
        self.device = device
        self.data_args = data_args
        self.image_retriever = image_retriever
        self.eval_dataset = dataset["dev_set"]
    
    @staticmethod
    def compute_recall(
            gold_url,
            image_urls,
            ids
        ):
        recall = 0
        for id in ids:
            pred_url = image_urls[id]
            if pred_url == gold_url:
                recall = 1

        return recall

    def run( 
            self,
        ):
        # load image urls
        print("... Loading images")
        images = self.image_retriever.load_images(
            device=self.device,
            dataset_path=self.data_args.dataset_path,
            encoding_path=self.data_args.encoding_path
        )
        _, image_urls = images
        
        # compute recall-1,5,10 averaged over dev set
        total_recall_1,  total_recall_5, total_recall_10 = [], [], []
        for sample in self.eval_dataset:
            text_sample, image_sample = sample[0], sample[1]
            probs_per_image = self.image_retriever.inference(
                self.device, text_sample, images
            )
            top_10 = torch.topk(probs_per_image, 10).indices
            recall_1_ids = top_10[:1]
            recall_5_ids = top_10[:5]
            recall_10_ids = top_10[:10]
            total_recall_1.append(self.compute_recall(image_sample, image_urls, recall_1_ids))
            total_recall_5.append(self.compute_recall(image_sample, image_urls, recall_5_ids))
            total_recall_10.append(self.compute_recall(image_sample, image_urls, recall_10_ids))
        
        recall_1 = sum(total_recall_1)/len(total_recall_1)
        recall_5 = sum(total_recall_5)/len(total_recall_5)
        recall_10 = sum(total_recall_10)/len(total_recall_10)

        print(
            f"... Metrics\n"
            f"> eval/recall-1\n{recall_1}\n"
            f"> eval/recall-5\n{recall_5}\n"
            f"> eval/recall-10\n{recall_10}"
        )


class ResponseGeneratorEvaluator():

    def __init__(
            self,
            training_args,
            response_generator,
            dataset,
            collator
        ):
        self.evaluator = Trainer(
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
        self.evaluator.evaluate()