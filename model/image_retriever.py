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
import numpy as np

from util.text import join_dialog
from util.image import load_image_from_url
from util.resource import get_device_util


LOGIT_SCALE_INIT_VALUE = 2.6592


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

    def load_images(
            self,
            device,
            raw_path,
            processed_path
        ):
        # load processed images
        if os.path.exists(processed_path):
            images = torch.load(processed_path)
        
        # iterate through raw dataset and save processed image embeddings and urls
        else:
            image_urls_unfiltered, image_urls, image_embeds = [], [], []
            file_path_list = sorted(glob.glob(raw_path+"/*/**"))
            for file_path in file_path_list:
                with open(file_path) as f:
                    data = json.load(f)
                image_urls_unfiltered.extend(d["photo_url"] for d in data)
            for url in image_urls_unfiltered:
                image = load_image_from_url(url)
                if image:
                    image_urls.append(url)
                    pixel = self.processor(images=image, return_tensors="pt").pixel_values
                    feature = self.model.get_image_features(pixel.to(device))
                    image_embeds.append(feature.detach().cpu())
            images = [image_embeds, image_urls]
            torch.save(images, processed_path)
        
        return images


    def inference(
            self,
            device,
            context,
            response,
            images
        ):
        # get text embeddings
        context.append(response)
        text = join_dialog(context, self.tokenizer.sep_token)
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True)
        text_embeds = self.model.get_text_features(text_inputs.input_ids.to(device))

        # get image embeddings
        image_embeds, image_urls = images
        image_embeds = torch.cat(image_embeds, dim=0).to(device)

        # normalized embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = nn.Parameter(torch.ones([]) * LOGIT_SCALE_INIT_VALUE).exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        # get top1 image
        probs_per_image = F.softmax(logits_per_image.squeeze(), dim=0)
        idx = torch.argmax(probs_per_image, dim=-1).item()
        image_url = image_urls[idx]

        return image_url