from dataclasses import dataclass


@dataclass
class ServeConfig(object):
    seed: int = 1234


@dataclass
class ImageRetrieverConfig(object):
    model_path: str = "/mnt/16tb/minyoung/checkpoints/photochat/bert_vit/checkpoint-6400"


@dataclass
class ResponseGeneratorConfig(object):
    model_path: str = "/mnt/16tb/minyoung/checkpoints/photochat/dialogpt_large/checkpoint-12000"


@dataclass
class DataConfig(object):
    images_dataset_path: str = "/mnt/16tb/minyoung/code/photochat/dataset/photochat"
    images_encoding_path: str = "/mnt/16tb/minyoung/code/photochat/dataset/photochat/image_encodings.pt"