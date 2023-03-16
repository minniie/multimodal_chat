import re
from typing import List

import torch
import numpy as np


ALLOWED_CHARS = re.compile('[A-Za-z0-9]+')


def truncate_dialog(context: List[str], max_context_len: int):
    return context[-max_context_len:]


def join_dialog(context: List[str], sep: str):
    return sep.join(context)


def split_dialog(context: str, sep: str):
    return context.split(sep)


def clean_uttr(dialog: List[str]):
    dialog = [re.sub(r'\s+', ' ', d).strip() for d in dialog]
    return dialog


def remove_empty_uttr(dialog: List[str]):
    dialog = list(filter(None, dialog))
    return dialog


def clean_decode(batch: List[int], tokenizer):
    batch = torch.where(batch < 0, tokenizer.pad_token_id, batch)
    text = tokenizer.batch_decode(np.expand_dims(batch, axis=-1), skip_special_tokens=True)
    text = [t.lower().strip() for t in text]
    text =  list(filter(None, text))
    # text = list(filter(lambda t: bool(ALLOWED_CHARS.match(t)), text))
    text = ['NONE'] if not text else text

    return text