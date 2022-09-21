import re
from typing import List


def truncate_context(
        context: List[str],
        max_context_len: int
    ):
    return context[-max_context_len:]


def join_context(
        context: List[str],
        sep: str
    ):
    return sep.join(context)


def split_context(
        context: str,
        sep: str
    ):
    return context.split(sep)


def strip_context(
        context: str,
        prefixes: List[str]
    ):
    for prefix in prefixes:
        context = context.replace(prefix, "")
    return context