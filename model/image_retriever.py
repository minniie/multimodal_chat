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
            text_model_name_or_path: str = "bert-base-uncased",
            image_model_name_or_path: str = "google/vit-base-patch16-224"
        ):
        self.text_model_name_or_path = text_model_name_or_path
        self.image_model_name_or_path = image_model_name_or_path
        self.load_processor()
        self.load_model()

    def load_processor(
            self
        ):
        tokenizer = BertTokenizer.from_pretrained(self.text_model_name_or_path)
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.image_model_name_or_path)
        self.processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)

    def load_model(
            self
        ):
        self.model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            self.image_model_name_or_path, self.text_model_name_or_path
        )