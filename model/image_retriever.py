import os
import json
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTFeatureExtractor,
    BertTokenizer,
)

from util.text import join_dialog
from util.image import load_image_from_url


LOGIT_SCALE_INIT_VALUE = 2.6592


class ImageRetriever():
    """
    Image retriever
    """

    def __init__(
            self,
            device,
            retriever_image_encoder_path: str = "google/vit-base-patch16-224",
            retriever_text_encoder_path: str = "bert-base-uncased",
            retriever_finetuned_path: str = None
        ):
        self.device = device
        self.retriever_image_encoder_path = retriever_image_encoder_path
        self.retriever_text_encoder_path = retriever_text_encoder_path
        self.retriever_finetuned_path = retriever_finetuned_path
        self.load_tokenizer()
        self.load_processor()
        self.load_model()

    def load_tokenizer(
            self
        ):
        self.tokenizer = BertTokenizer.from_pretrained(self.retriever_text_encoder_path)

    def load_processor(
            self
        ):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.retriever_image_encoder_path)
        self.processor = VisionTextDualEncoderProcessor(self.feature_extractor, self.tokenizer)

    def load_model(
            self
        ):
        if self.retriever_finetuned_path:
            self.model = VisionTextDualEncoderModel.from_pretrained(
                self.retriever_finetuned_path
            )
        else:
            self.model = VisionTextDualEncoderModel.from_vision_text_pretrained(
                self.retriever_image_encoder_path, self.retriever_text_encoder_path
            )
        self.model.to(self.device)

    def load_images(
            self,
            device,
            dataset_path,
            encoding_path
        ):
        # load image encodings
        if os.path.exists(encoding_path):
            images = torch.load(encoding_path)
        
        # iterate through raw dataset and save image encodings and urls
        else:
            image_urls_unfiltered, image_urls, image_encodings = [], [], []
            file_path_list = sorted(glob.glob(dataset_path+"/test/**"))
            for file_path in file_path_list:
                print(f"> file path\n{file_path}")
                with open(file_path) as f:
                    data = json.load(f)
                image_urls_unfiltered.extend(d["photo_url"] for d in data)
            for url in image_urls_unfiltered:
                image = load_image_from_url(url)
                if image:
                    image_urls.append(url)
                    pixel = self.processor(images=image, return_tensors="pt").pixel_values
                    feature = self.model.get_image_features(pixel.to(device))
                    image_encodings.append(feature.detach().cpu())
            images = [image_encodings, image_urls]
            torch.save(images, encoding_path)
        
        return images

    def inference(
            self,
            device,
            context,
            images
        ):

        """
        https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py#L296
        """
        
        # get text encodings
        text = join_dialog(context, self.tokenizer.sep_token)
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True)
        text_encodings = self.model.get_text_features(text_inputs.input_ids.to(device))

        # get image encodings
        image_encodings, image_urls = images
        image_encodings = torch.cat(image_encodings, dim=0).to(device)

        # normalized encodings
        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = nn.Parameter(torch.ones([]) * LOGIT_SCALE_INIT_VALUE).exp()
        logits_per_text = torch.matmul(text_encodings, image_encodings.t()) * logit_scale
        logits_per_image = logits_per_text.T
        probs_per_image = F.softmax(logits_per_image.squeeze(), dim=0)

        return probs_per_image