import torch
from torch.nn.utils.rnn import pad_sequence


class ImageRetrieverCollator():

    def __init__(self):
        pass


class ResponseGeneratorCollator():
    
    def __init__(
            self,
            tokenizer
        ):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        contexts = [self.tokenizer.eos_token.join(s[:-1]) + self.tokenizer.eos_token for s in samples]
        responses = [s[-1] + self.tokenizer.eos_token for s in samples]
        # print("==============")
        # print(contexts)
        # print("==============")
        # print(responses)
    
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
        # print(input_ids)
        # print(attention_mask)
        # print(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
      