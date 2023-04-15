from collections import Counter

import torch
from transformers import Trainer
from sklearn.metrics import f1_score

from learning.callback import ResponseGeneratorCallback


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
        self.eval_dataset = dataset["test_set"]
    
    @staticmethod
    def compute_recall(
            gold_image_url,
            image_urls,
            top_image_ids
        ):
        recall = 0
        for id in top_image_ids:
            pred_url = image_urls[id]
            if pred_url == gold_image_url:
                recall = 1

        return recall

    @staticmethod
    def compute_mrr(
            gold_image_url,
            image_urls,
            sorted_image_ids
        ):
        if gold_image_url in image_urls:
            gold_image_id = image_urls.index(gold_image_url)
            r = (sorted_image_ids == gold_image_id).nonzero().item()
            mrr = 1 / (r + 1)
        else:
            mrr = None
        
        return mrr

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

        # compute automatic evaluation metrics
        print("... Computing metrics")
        cls_true, cls_pred = [], [] 
        total_recall_1,  total_recall_5, total_recall_10 = [], [], []
        total_mrr = []
        for i, sample in enumerate(self.eval_dataset):
            dialogue_history, gold_image_url = sample[0], sample[1]
            probs_per_image = self.image_retriever.inference(
                self.device, dialogue_history, images
            )

            # compute classification f1
            top_10_values = torch.topk(probs_per_image, 10).values
            if gold_image_url == "":
                cls_true.append(0)
            else:
                cls_true.append(1)
            if top_10_values[0].item() > 0.05:
                cls_pred.append(1)
            else:
                cls_pred.append(0)
  
            # compute recall-1,5,10
            if gold_image_url != "":
                top_10_image_ids = torch.topk(probs_per_image, 10).indices
                total_recall_1.append(self.compute_recall(gold_image_url, image_urls, top_10_image_ids[:1]))
                total_recall_5.append(self.compute_recall(gold_image_url, image_urls, top_10_image_ids[:5]))
                total_recall_10.append(self.compute_recall(gold_image_url, image_urls, top_10_image_ids[:10]))
                
            # compute mrr
            if gold_image_url != "":
                sorted_image_ids = torch.topk(probs_per_image, probs_per_image.size(-1)).indices
                mrr = self.compute_mrr(gold_image_url, image_urls, sorted_image_ids)
                if mrr:
                    total_mrr.append(mrr)

            # print example dialogue history and top 5 ranked images
            if gold_image_url != "" and i % 50 == 0 and total_recall_5[-1] == 1:
                print(
                    f"... Examples\n"
                    f"> dialogue history\n{dialogue_history}\n"
                    f"> ground-truth image\n{gold_image_url}\n"
                    f"> top 5 images\n{[image_urls[id] for id in top_10_image_ids[:5]]}"
                )
      
        cls_f1 = f1_score(cls_true, cls_pred)
        cnt_true = Counter(cls_true)
        cnt_pred = Counter(cls_pred)
        recall_1 = sum(total_recall_1)/len(total_recall_1)
        recall_5 = sum(total_recall_5)/len(total_recall_5)
        recall_10 = sum(total_recall_10)/len(total_recall_10)
        mrr = sum(total_mrr)/len(total_mrr)

        print(
            f"... Results\n"
            f"> eval/cls-f1\n{cls_f1}\n"
            f"> # true/pred valid images\n{cnt_true[1]}/{cnt_pred[1]}\n"
            f"> # true/pred dummy images\n{cnt_true[0]}/{cnt_pred[0]}\n"
            f"> eval/recall-1\n{recall_1}\n"
            f"> eval/recall-5\n{recall_5}\n"
            f"> eval/recall-10\n{recall_10}\n"
            f"> eval/mrr\n{mrr}"
        )


class ResponseGeneratorEvaluator():

    def __init__(
            self,
            training_args,
            response_generator,
            dataset,
            collator
        ):
        # use huggingface trainer as evaluation mode
        self.evaluator = Trainer(
            args=training_args,
            model=response_generator.model,
            tokenizer=response_generator.tokenizer,
            train_dataset=dataset["train_set"],
            eval_dataset=dataset["test_set"],
            data_collator=collator,
            callbacks=[ResponseGeneratorCallback]
        )

    def run(
            self,
        ):
        self.evaluator.evaluate()