from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)


@dataclass
class ModelArguments:
    """
    Arguments for which model will be trained or evaluated
    """
    image_model_name: str = field(
        default=None,
        metadata={
            "help": "image model name for image retriever"
        }
    )
    text_model_name: str = field(
        default=None,
        metadata={
            "help": "text model name for image retriever"
        }
    )
    image_text_model_name: str = field(
        default=None,
        metadata={
            "help": "VisionTextDualEncoderModel name for image retriever"
        }
    )
    generator_model_name: str = field(
        default=None,
        metadata={
            "help": "generator model name for response generator"
        }
    )
    use_image_as_generator_input: bool = field(
        default=False,
        metadata={
            "help": "whether to use image as input to response generator"
        }
    )


@dataclass
class DataArguments:
    """
    Arguments for dataset to train and evaluate the models
    """
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "path to dataset for training models"
        }
    )
    encoding_path: str = field(
        default=None,
        metadata={
            "help": "path to dataset preprocessed as encodings"
        }
    )
    max_seq_len: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization."
            "Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )


def set_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args,training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    return model_args, data_args, training_args