from dataclasses import dataclass


@dataclass
class ServeConfig(object):
    seed: int = 1234


@dataclass
class ImageRetrieverConfig(object):
    pass


@dataclass
class ResponseGeneratorConfig(object):
    generator_model_path: str = "/mnt/16tb/minyoung/checkpoints/photochat/dummy/checkpoint-140"