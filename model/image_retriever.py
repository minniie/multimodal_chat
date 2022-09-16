from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTFeatureExtractor,
    BertTokenizer,
)


class ImageRetriever():
    """
    Image retriever
    """

    def __init__(
            self,
            text_model_name: str = "bert-base-uncased",
            image_model_name: str = "google/vit-base-patch16-224"
        ):
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name

    def load_processor(
            self
        ):
        tokenizer = BertTokenizer.from_pretrained(self.text_model_name)
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.image_model_name)
        self.processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)

    def load_model(
            self
        ):
        self.model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            self.image_model_name, self.text_model_name
        )