from unittest.mock import patch
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import ToTensor

from util.image import load_image_from_url, create_dummy_image
from util.text import join_dialog


class ImageRetrieverCollator():

    def __init__(
            self,
            tokenizer,
            processor
        ):
        self.tokenizer = tokenizer
        self.processor = processor

    def __call__(self, samples):
        images, texts = [], []
        for sample in samples:
            text_sample, image_sample = sample[0], sample[1]
            image = load_image_from_url(image_sample)
            if image:
                images.append(image)
                texts.append(join_dialog(text_sample, self.tokenizer.sep_token))
        
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", 
            padding="max_length", truncation=True, max_length=512
        )
        
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "return_loss": True,
        }


class ResponseGeneratorCollator():
    
    def __init__(
            self,
            tokenizer,
            processor,
            use_image_as_generator_input
        ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.use_image_as_generator_input = use_image_as_generator_input
    
    def _batchify_text_and_image_to_text(
            self,
            samples
        ):
        contexts, responses, images = [], [], []
        for sample in samples:
            text_sample, image_sample = sample[0], sample[1]
            context = join_dialog(text_sample[:-1], self.tokenizer.sep_token)
            response = text_sample[-1]
            if self.use_image_as_generator_input:
                image = load_image_from_url(image_sample)
            if not image:
                image = create_dummy_image()
            contexts.append(context)
            responses.append(response)
            images.append(image)

        inputs = self.processor(
            text=contexts, images=images, return_tensors="pt",
            padding="max_length", truncation=True, max_length=512
        )
        labels = self.processor(
            text=responses, return_tensors="pt",
            padding="max_length", truncation=True, max_length=512
        )

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "labels": labels.input_ids
        }

    def _batchify_text_to_text(
            self,
            samples
        ):
        contexts, responses = [], []
        for sample in samples:
            text_sample = sample[0]
            context = join_dialog(text_sample[:-1], self.tokenizer.sep_token) + self.tokenizer.sep_token
            response = text_sample[-1] + self.tokenizer.eos_token
            contexts.append(context)
            responses.append(response)
    
        input_ids, labels = [], []
        for context, response in zip(contexts, responses):
            context_ids = self.tokenizer.encode(context)
            response_ids = self.tokenizer.encode(response)
            input_ids.append(torch.LongTensor(context_ids + response_ids))
            labels.append(torch.LongTensor(len(context_ids)*[-100] + response_ids))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def __call__(self, samples):
        if self.use_image_as_generator_input:
            return self._batchify_text_and_image_to_text(samples)
        else:
            return self._batchify_text_to_text(samples)