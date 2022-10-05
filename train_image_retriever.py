from dataset.processor import PhotochatProcessor
from dataset.collator import ImageRetrieverCollator
from model.image_retriever import ImageRetriever
from learning.trainer import ImageRetrieverTrainer
from util.args import set_args
from util.resource import set_device, get_device_util


def main():
    # set arguments
    model_args, data_args, training_args = set_args()

    # set device
    device = set_device()

    # load model
    print(f"{'*'*10} Loading model")
    image_retriever = ImageRetriever(device, model_args.text_model_name, model_args.image_model_name)
    
    # get device util
    print(f"{'*'*10} Device util after loading model")
    get_device_util()

    # load dataset and collator
    print(f"{'*'*10} Loading dataset")
    p = PhotochatProcessor()
    p.split(data_args.dataset_path)
    dataset = p.data_for_image_retriever
    collator = ImageRetrieverCollator(image_retriever.processor, image_retriever.tokenizer)

    # set trainer
    trainer = ImageRetrieverTrainer(
        training_args,
        image_retriever,
        dataset,
        collator
    )
    trainer.train()


if __name__ == "__main__":
    main()