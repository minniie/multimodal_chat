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
    
    def run( 
            self,
        ):
        
        # load images: TODO
        print("... Loading images")
        images = self.image_retriever.load_images(
            device=self.device,
            dataset_path=self.data_args.dataset_path,
            encoding_path=self.data_args.encoding_path
        )
        for sample in self.eval_dataset:
            text_sample, image_sample = sample[0], sample[1]
            image = load_image_from_url(image_sample)
            if not image:
                continue
            image_encoding = self.image_retriever.processor(
                image=image, return_tensors="pt"
            )
            probs_per_image = self.image_retriever.inference(
                self.device, text_sample, images
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