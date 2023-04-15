from dataclasses import dataclass


@dataclass
class ServeConfig(object):
    seed: int = 1234


@dataclass
class ImageRetrieverConfig(object):
    image_encoder_path: str = "google/vit-base-patch16-224"
    text_encoder_path: str = "bert-base-uncased"
    finetuned_path: str = "/mnt/16tb/minyoung/checkpoints/photochat/image_retriever/vit_base_bert_base/checkpoint-6430"
    use_model: bool = True


@dataclass
class ResponseGeneratorConfig(object):
    image_encoder_path: str = "google/vit-large-patch32-384"
    text_decoder_path: str = "microsoft/DialoGPT-medium"
    finetuned_path: str = "/mnt/16tb/minyoung/checkpoints/photochat/response_generator/vit_large_dialogpt_medium/checkpoint-11174"
    # image_encoder_path: str = None
    # text_decoder_path: str = "microsoft/DialoGPT-medium"
    # finetuned_path: str = "/mnt/16tb/minyoung/checkpoints/photochat/response_generator/dialogpt_medium/checkpoint-11000"


@dataclass
class DataConfig(object):
    images_dataset_path: str = "/mnt/16tb/minyoung/code/photochat/dataset/photochat"
    images_encoding_path: str = "/mnt/16tb/minyoung/code/photochat/dataset/photochat/image_encodings.pt"