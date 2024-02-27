"""
Misc utility functions. mattma1970@gmail
"""

import streamlit as st
from typing import List
import base64
import numpy as np
import av
from enum import Enum
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


def time_only(time: datetime) -> int:
    return time.hour * 60 + time.minute


def turn_sum(chat_history_lengths: List[List[int]]) -> int:
    """
    Chat history lengths are stored as a List[List[int]] eg. [[19,200],[50,234]] where
    each inner list is the list of lengths, in tokens, of the content from system/user/assistant.
    A 'turn' refers to a turn taken by one of these roles.
    """
    return sum(sum(a) for a in chat_history_lengths)


def endpoint(root_url: str, func: str):
    return "/".join([root_url.strip("/"), func])


def remove_problem_chars(
    text: str, acceptable_chars_pattern: str = "[^a-zA-Z0-9,. \s'?!;:]"
):
    """
    Remove chars from text. Used in pre-processing text input to STT.
    Args:
        text: text to be filtered
        accpetable_chars_pattern: str: a valid regex pattern for acceptable characters. All other will be filtered.
    """
    if acceptable_chars_pattern is None or acceptable_chars_pattern == "":
        return text

    pattern = re.compile(acceptable_chars_pattern)
    ret = text
    try:
        ret = pattern.sub(lambda match: "", text)
        ret = re.sub(r"\s+", " ", ret)
    except Exception as e:
        logger.error(e)
    return ret


def has_pronouns(text: str, additional_words: List[str] = []) -> bool:
    pronouns: List[str] = ["it", "its", "it's", "they"]
    pronouns.extend(additional_words)
    pronouns_check = [
        bool(re.search(r"\b" + re.escape(pronoun) + r"\b", text))
        for pronoun in pronouns
    ]
    return any(pronouns_check)


def safe_key(key):
    return re.sub(r"[^0-9a-zA-Z]", "", key)
