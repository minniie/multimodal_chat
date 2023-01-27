import math

from transformers.integrations import TensorBoardCallback

from util.metric import BLEU


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
            labels = inputs.pop("labels")
            outputs = model.generate(**inputs)
            pred_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            label_batch = tokenizer.batch_decode(labels, skip_special_tokens=True)
            pred_text = [p.split() for p in pred_batch]
            label_text = [[l.split()] for l in label_batch]
            preds_text.extend(pred_text)
            labels_text.extend(label_text)
        
        bleu = BLEU(preds_text, labels_text)

        self.tb_writer.add_scalar("eval/ppl", ppl, state.global_step)
        self.tb_writer.add_scalar("eval/bleu-1", bleu["bleu-1"], state.global_step)
        self.tb_writer.add_scalar("eval/bleu-2", bleu["bleu-2"], state.global_step)