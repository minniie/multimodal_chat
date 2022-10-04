import warnings
import requests

from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence


IMAGE_DIM = 224


class ImageRetrieverCollator():

    def __init__(
            self,
            processor,
            tokenizer
        ):
        self.processor = processor
        self.tokenizer = tokenizer

    def __call__(self, samples):
        # exclude images with any exceptions or warnings
        warnings.simplefilter("error")
        images, text = [], []
        print(samples)
        # text = [self.tokenizer.sep_token.join(s[0]) for s in samples]
        for s in samples:
            try:
                image = Image.open(requests.get(s[1], stream=True).raw).convert('RGB').resize((IMAGE_DIM, IMAGE_DIM))
                images.append(image)
                text.append(self.tokenizer.sep_token.join(s[0]))
            except Exception as e:
                print(f"Exception: {e}")
                continue
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
      