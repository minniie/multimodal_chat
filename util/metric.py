import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from lexical_diversity import lex_div as ld


def Perplexity(logits, labels):
    loss_fn = nn.CrossEntropyLoss()
    logits_batch = torch.tensor(logits).transpose(1, 2)
    labels_batch = torch.tensor(labels)
    loss = loss_fn(logits_batch, labels_batch)
    ppl = torch.exp(loss)
    
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


def DistinctN(preds):
    def bigram(pred):
        return [" ".join(pred[i:i+2]) for i in range(len(pred)-1)]
    preds_unigram = preds
    preds_bigram = list(map(bigram, preds))
    distinct_1 = list(map(ld.ttr, preds_unigram))
    distinct_1 = sum(distinct_1)/len(distinct_1)
    distinct_2 = list(map(ld.ttr, preds_bigram))
    distinct_2 = sum(distinct_2)/len(distinct_2)

    return {
        "distinct-1": distinct_1,
        "distinct-2": distinct_2
    }