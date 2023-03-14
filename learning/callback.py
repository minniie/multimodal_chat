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
        
        # calculate bleu
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        eval_dataloader = kwargs["eval_dataloader"]
        preds_text, labels_text = [], []
        for batch in eval_dataloader:
            for sample in zip(batch["input_ids"], batch["pixel_values"], batch["labels"]):
                input_ids = sample[0][sample[0] > 0].unsqueeze(0).to(model.device)
                pixel_values = sample[1].unsqueeze(0).to(model.device)
                label = sample[2]
                pred = model.generate(
                    input_ids=input_ids, pixel_values=pixel_values,
                    max_length=64, num_beams=1, do_sample=True
                ).squeeze().to("cpu")
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