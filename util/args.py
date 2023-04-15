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
    retriever_image_encoder_path: str = field(
        default=None,
        metadata={
            "help": "path for image encoder of image retriever"
        }
    )
    retriever_text_encoder_path: str = field(
        default=None,
        metadata={
            "help": "path for text encoder of image retriever"
        }
    )
    retriever_finetuned_path: str = field(
        default=None,
        metadata={
            "help": "path for finetuned checkpoint of image retriever"
        }
    )
    generator_image_encoder_path: str = field(
        default=None,
        metadata={
            "help": "path for image encoder of response generator"
        }
    )
    generator_text_decoder_path: str = field(
        default=None,
        metadata={
            "help": "path for text decoder of response generator"
        }
    )
    generator_finetuned_path: str = field(
        default=None,
        metadata={
            "help": "path for finetuned checkpoint of response generator"
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