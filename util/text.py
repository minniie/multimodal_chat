import re
from typing import List


def truncate_context(context: List[str], max_context_len: int):
    return context[-max_context_len:]


def join_context(context: List[str], sep: str):
    return sep.join(context)


def split_context(context: str, sep: str):
    return context.split(sep)


def clean_uttr(dialog: List[str]):
    dialog = [re.sub(r'\s+', ' ', d).strip() for d in dialog]
    return dialog


def remove_empty_uttr(dialog: List[str]):
    dialog = list(filter(None, dialog))
    return dialog