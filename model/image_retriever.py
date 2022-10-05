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
            device,
            text_model_name_or_path: str = "bert-base-uncased",
            image_model_name_or_path: str = "google/vit-base-patch16-224",
            multimodal_model_name_or_path: str = None
        ):
        self.device = device
        self.text_model_name_or_path = text_model_name_or_path
        self.image_model_name_or_path = image_model_name_or_path
        self.multimodal_model_name_or_path = multimodal_model_name_or_path
        self.load_processor()
        self.load_model()

    def load_processor(
            self
        ):
        self.tokenizer = BertTokenizer.from_pretrained(self.text_model_name_or_path)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.image_model_name_or_path)
        self.processor = VisionTextDualEncoderProcessor(self.feature_extractor, self.tokenizer)

    def load_model(
            self
        ):
        if self.multimodal_model_name_or_path:
            self.model = VisionTextDualEncoderModel.from_pretrained(
                self.multimodal_model_name_or_path
            )
        else:
            self.model = VisionTextDualEncoderModel.from_vision_text_pretrained(
                self.image_model_name_or_path, self.text_model_name_or_path
            )
        self.model.to(self.device)

    def inference(
            self
        ):
        pass