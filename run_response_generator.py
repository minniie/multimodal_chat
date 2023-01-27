from dataset.processor import PhotochatProcessor
from dataset.collator import ResponseGeneratorCollator
from model.response_generator import ResponseGenerator
from learning.trainer import ResponseGeneratorTrainer
from util.args import set_args
from util.resource import set_device, get_device_util


def main():
    # set arguments
    model_args, data_args, task_args, training_args = set_args()

    # set device
    device = set_device()

    # load model
    print("... Loading model")
    response_generator = ResponseGenerator(
        device,
        model_args.generator_model_name,
        model_args.use_image_as_generator_input
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
        response_generator.tokenizer,
        response_generator.processor,
        model_args.use_image_as_generator_input
    )

    # set trainer
    trainer = ResponseGeneratorTrainer(
        training_args,
        response_generator,
        dataset,
        collator,
        model_args.use_image_as_generator_input
    )
    trainer.run(task_args)


if __name__ == "__main__":
    main()