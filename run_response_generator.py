from dataset.processor import PhotochatProcessor
from dataset.collator import ResponseGeneratorCollator
from model.response_generator import ResponseGenerator
from learning.trainer import ResponseGeneratorTrainer
from learning.evaluator import ResponseGeneratorEvaluator
from util.args import set_args
from util.resource import set_device, get_device_util


def main():
    # set arguments
    model_args, data_args, training_args = set_args()

    # set device
    device = set_device()

    # load model
    print("... Loading model")
    response_generator = ResponseGenerator(
        device,
        model_args.generator_image_encoder_path,
        model_args.generator_text_decoder_path
    )

    # get device util
    print(f"... Device util after loading model")
    get_device_util()

    # load dataset and collator
    print(f"... Loading dataset")
    processor = PhotochatProcessor()
    processor.split(data_args.dataset_path)
    dataset = processor.data_for_response_generator
    collator = ResponseGeneratorCollator(
        response_generator.user_token,
        response_generator.bot_token,
        response_generator.tokenizer,
        response_generator.processor
    )

    # set trainer or evaluator
    if training_args.do_train:
        trainer = ResponseGeneratorTrainer(
            training_args,
            response_generator,
            dataset,
            collator
        )
        trainer.run()
    if training_args.do_eval:
        evaluator = ResponseGeneratorEvaluator(
            training_args,
            response_generator,
            dataset,
            collator
        )
        evaluator.run()


if __name__ == "__main__":
    main()