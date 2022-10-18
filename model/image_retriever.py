import torch
import torch.nn.functional as F
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTFeatureExtractor,
    BertTokenizer,
)
import numpy as np

from util.text import join_dialog


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
            self,
            device,
            context,
            response,
            images
        ):
        context.append(response)
        text = join_dialog(context, self.tokenizer.sep_token)
        print("======= input text =======")
        print(text)
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        images_pixel, images_url = images
        outputs = self.model(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            pixel_values=images_pixel.to(device)
        )
        logits_per_image = outputs.logits_per_image
        probs_per_image = F.softmax(logits_per_image.squeeze(), dim=0)
        idx = torch.argmax(probs_per_image, dim=-1).item()
        image_url = images_url[idx]

        return image_url