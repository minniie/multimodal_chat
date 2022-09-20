from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval
    """
    max_seq_len: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization."
            "Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from
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


def set_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    return model_args, data_args, training_args