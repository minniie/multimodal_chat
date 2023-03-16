from dataset.processor import PhotochatProcessor
from dataset.collator import ImageRetrieverCollator
from model.image_retriever import ImageRetriever
from learning.trainer import ImageRetrieverTrainer
from learning.evaluator import ImageRetrieverEvaluator
from util.args import set_args
from util.resource import set_device, get_device_util


def main():
    # set arguments
    model_args, data_args, training_args = set_args()

    # set device
    device = set_device()

    # load model
    print("... Loading model")
    image_retriever = ImageRetriever(
        device,
        model_args.image_model_name,
        model_args.text_model_name,
        model_args.image_text_model_name
    )
    
    # get device util
    print("... Device util after loading model")
    get_device_util()

    # load dataset and collator
    print("... Loading dataset")
    processor = PhotochatProcessor()
    processor.split(data_args.dataset_path)
    dataset = processor.data_for_image_retriever
    collator = ImageRetrieverCollator(
        image_retriever.tokenizer,
        image_retriever.processor
    )

    # set trainer or evaluator
    if training_args.do_train:
        trainer = ImageRetrieverTrainer(
            training_args,
            image_retriever,
            dataset,
            collator
        )
        trainer.run()
    if training_args.do_eval:
        evaluator = ImageRetrieverEvaluator(
            device,
            data_args,
            image_retriever,
            dataset
        )
        evaluator.run()


if __name__ == "__main__":
    main()