import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu


def Perplexity(logits, labels):
    loss_fn = nn.CrossEntropyLoss()
    logits_batch = torch.tensor(logits).transpose(1, 2)
    labels_batch = torch.tensor(labels)
    loss = loss_fn(logits_batch, labels_batch)
    ppl = torch.exp(loss).item()
    
    return {
        "ppl": ppl
    }


def BLEU(preds, labels):
    bleu_1 = corpus_bleu(labels, preds, weights=(1,0,0,0))
    bleu_2 = corpus_bleu(labels, preds, weights=(0,1,0,0))

    return {
        "bleu-1": bleu_1,
        "bleu-2": bleu_2
    }


def DistinctN(preds, labels):
    # TODO
    distinct_1 = 0.0
    distinct_2 = 0.0

    return {
        "distinct-1": distinct_1,
        "distinct-2": distinct_2
    }