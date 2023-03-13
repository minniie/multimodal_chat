import math

from transformers.integrations import TensorBoardCallback

from util.metric import BLEU, DistinctN
from util.text import normalize_decode_per_token


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
        for inputs in eval_dataloader:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = inputs.pop("labels").to("cpu")
            preds = model.generate(**inputs).to("cpu")
            for pred, label in zip(preds, labels):
                preds_text.append(normalize_decode_per_token(pred, tokenizer))
                labels_text.append([normalize_decode_per_token(label, tokenizer)])
        
        # calculate bleu and distinct-n
        bleu = BLEU(preds_text, labels_text)
        distinct_n = DistinctN(preds_text)

        # report to tensorboard
        print(bleu, distinct_n)
        self.tb_writer.add_scalar("eval/ppl", ppl, state.global_step)
        self.tb_writer.add_scalar("eval/bleu-1", bleu["bleu-1"], state.global_step)
        self.tb_writer.add_scalar("eval/bleu-2", bleu["bleu-2"], state.global_step)
        self.tb_writer.add_scalar("eval/distinct-1", distinct_n["distinct-1"], state.global_step)
        self.tb_writer.add_scalar("eval/distinct-2", distinct_n["distinct-2"], state.global_step)