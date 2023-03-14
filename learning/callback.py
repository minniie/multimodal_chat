import math

import torch
from transformers.integrations import TensorBoardCallback

from util.metric import BLEU, DistinctN
from util.text import batch_decode


class MetricCallback(TensorBoardCallback):
    
    def on_evaluate(self, args, state, control, **kwargs):
        # calculate ppl
        loss = state.log_history[-1]["eval_loss"]
        ppl = math.exp(loss)
        
        # get model and dataset
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        eval_dataloader = kwargs["eval_dataloader"]
        preds_text, labels_text = [], []

        for batch in eval_dataloader:
            # generate with text and image inputs
            if "pixel_values" in batch.keys():
                for sample in zip(batch["input_ids"], batch["pixel_values"], batch["labels"]):
                    # remove all elements of [PAD] (right padding)
                    input_ids = sample[0][sample[0] > 0].unsqueeze(0).to(model.device)
                    pixel_values = sample[1].unsqueeze(0).to(model.device)
                    label = sample[2]
                    pred = model.generate(
                        input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=64
                    ).squeeze().to("cpu")
                    preds_text.append(batch_decode(pred, tokenizer))
                    labels_text.append([batch_decode(label, tokenizer)])
        
            # genenrate with text input only
            else:
                for sample in zip(batch["input_ids"], batch["labels"]):
                    # remove all elements of <|endoftext|> (right padding) and -100 (left padding)
                    input_mask = torch.logical_and(sample[0] != sample[1], sample[0] != tokenizer.pad_token_id)
                    input_ids = sample[0][input_mask].unsqueeze(0).to(model.device)
                    label = sample[1]
                    pred = model.generate(input_ids=input_ids, max_new_tokens=64).squeeze().to("cpu")
                    pred = pred[input_ids.size(-1):]
                    preds_text.append(batch_decode(pred, tokenizer))
                    labels_text.append([batch_decode(label, tokenizer)])
        
        # calculate bleu and distinct-n
        bleu = BLEU(preds_text, labels_text)
        distinct_n = DistinctN(preds_text)

        # report to tensorboard
        self.tb_writer.add_scalar("eval/ppl", ppl, state.global_step)
        self.tb_writer.add_scalar("eval/bleu-1", bleu["bleu-1"], state.global_step)
        self.tb_writer.add_scalar("eval/bleu-2", bleu["bleu-2"], state.global_step)
        self.tb_writer.add_scalar("eval/distinct-1", distinct_n["distinct-1"], state.global_step)
        self.tb_writer.add_scalar("eval/distinct-2", distinct_n["distinct-2"], state.global_step)