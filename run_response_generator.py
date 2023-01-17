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
    print(f"{'*'*10} Loading model")
    response_generator = ResponseGenerator(device, model_args.generator_model_name)

    # get device util
    print(f"{'*'*10} Device util after loading model")
    get_device_util()

    # load dataset and collator
    print(f"{'*'*10} Loading dataset")
    p = PhotochatProcessor()
    p.split(data_args.dataset_path)
    dataset = p.data_for_response_generator
    collator = ResponseGeneratorCollator(response_generator.tokenizer)

    # set trainer
    trainer = ResponseGeneratorTrainer(
        training_args,
        response_generator,
        dataset,
        collator
    )
    trainer.run(task_args)


if __name__ == "__main__":
    main()