from dataset.preprocessor import PhotochatPreprocessor
from dataset.collator import ResponseGeneratorCollator
from model.response_generator import ResponseGenerator
from learning.trainer import ResponseGeneratorTrainer
from util.args import set_args


def main():
    # set arguments
    model_args, data_args, training_args = set_args()

    # load model
    print(f"{'*'*10} Loading model")
    response_generator = ResponseGenerator(model_args.generator_model_name)

    # load dataset and collator
    print(f"{'*'*10} Loading dataset")
    p = PhotochatPreprocessor()
    dataset = p.data_for_response_generator
    collator = ResponseGeneratorCollator(response_generator.tokenizer)

    # set trainer
    trainer = ResponseGeneratorTrainer(
        training_args,
        response_generator,
        dataset,
        collator
    )
    trainer.train()


if __name__ == "__main__":
    main()