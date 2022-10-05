import warnings

import torch
from torch.nn.utils.rnn import pad_sequence

from util.image import load_image_from_url


class ImageRetrieverCollator():

    def __init__(
            self,
            processor,
            tokenizer
        ):
        self.processor = processor
        self.tokenizer = tokenizer

    def __call__(self, samples):
        images, text = [], []
        for s in samples:
            image = load_image_from_url(s[1])
            if image:
                images.append(image)
                text.append(self.tokenizer.sep_token.join(s[0]))
        
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", 
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
            tokenizer
        ):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        contexts = [self.tokenizer.eos_token.join(s[:-1]) + self.tokenizer.eos_token for s in samples]
        responses = [s[-1] + self.tokenizer.eos_token for s in samples]
    
        input_ids, labels = [], []
        for context, response in zip(contexts, responses):
            context_ids = self.tokenizer.encode(context)
            response_ids = self.tokenizer.encode(response)
            input_ids.append(torch.LongTensor(
                context_ids + response_ids
            ))
            labels.append(torch.LongTensor(
                len(context_ids)*[-100] + response_ids
            ))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
      