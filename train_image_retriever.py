from dataset.preprocessor import PhotochatPreprocessor
from dataset.collator import ImageRetrieverCollator
from model.image_retriever import ImageRetriever
from learning.trainer import ImageRetrieverTrainer
from util.args import set_args


def main():
    # set arguments
    model_args, data_args, training_args = set_args()

    # load model
    # print(f"{'*'*10} Loading model")
    # image_retriever = ImageRetriever(model_args.text_model_name, model_args.image_model_name)

        # load dataset and collator
    print(f"{'*'*10} Loading dataset")
    p = PhotochatPreprocessor()
    dataset = p.data_for_image_retriever
    collator = ImageRetrieverCollator(image_retriever.processor)

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