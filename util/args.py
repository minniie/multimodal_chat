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
    text_model_name: str = field(
        default=None,
        metadata={
            "help": "text model name for image retriever"
        }
    )
    image_model_name: str = field(
        default=None,
        metadata={
            "help": "image model name for image retriever"
        }
    )
    generator_model_name: str = field(
        default=None,
        metadata={
            "help": "generator model name for response generator"
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
            "help": "dataset path to train models"
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


@dataclass
class TaskArguments:
    """
    Arguments for which task the script is used for
    """
    task: str = field(
        default=None,
        metadata={
            "help": "training or evaluation"
        }
    )


def set_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TaskArguments, TrainingArguments))
    model_args, data_args, task_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    return model_args, data_args, task_args, training_args