import torch
from torch.nn.utils.rnn import pad_sequence

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
            user_token,
            bot_token,
            tokenizer,
            processor,
            use_image_as_generator_input
        ):
        self.user_token = user_token
        self.bot_token = bot_token
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
            text_sample_prefixed = text_sample.copy()
            for id in range(len(text_sample)):
                if (len(text_sample) % 2 == 0 and id % 2 == 0) \
                    or (len(text_sample) % 2 == 1 and id % 2 == 1):
                    text_sample_prefixed[id] = self.user_token + text_sample[id]
                else:
                    text_sample_prefixed[id] = self.bot_token + text_sample[id]
            context = join_dialog(text_sample_prefixed[:-1], "")
            response = text_sample_prefixed[-1][len(self.bot_token):]
            image = load_image_from_url(image_sample)
            if not image:
                image = create_dummy_image()
            contexts.append(context)
            responses.append(response)
            images.append(image)

        # automatically adds [CLS] at beginning and [SEP] at end
        inputs = self.processor(
            text=contexts, images=images, return_tensors="pt",
            padding="max_length", truncation=True, max_length=512
        )
        labels = self.processor(
            text=responses, return_tensors="pt",
            padding="max_length", truncation=True, max_length=512
        )
        labels.input_ids = labels.input_ids[:, 1:]  # remove CLS

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
            text_sample_prefixed = text_sample.copy()
            for id in range(len(text_sample)):
                if (len(text_sample) % 2 == 0 and id % 2 == 0) \
                    or (len(text_sample) % 2 == 1 and id % 2 == 1):
                    text_sample_prefixed[id] = self.user_token + text_sample[id]
                else:
                    text_sample_prefixed[id] = self.bot_token + text_sample[id]
            context = self.tokenizer.bos_token + join_dialog(text_sample_prefixed[:-1], "") + self.bot_token
            response = text_sample_prefixed[-1][len(self.bot_token):] + self.tokenizer.eos_token
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
        labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

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