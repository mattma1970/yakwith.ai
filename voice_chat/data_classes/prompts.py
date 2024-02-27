""" 
Class to manage the accumulation of prompt fragments until a sensible whole is found
"""

from typing import Optional, Dict, Union, List, Any
from os import PathLike
import json
import re
from attr import define, Factory, field
import logging
from datetime import datetime
from datetime import timedelta


@define
class PromptAccumulator:
    prompt_accumulator: list[Dict] = field(factory=list)
    TTL_in_seconds: int = field(
        default=20
    )  # number of seconds after which a chunk of prompt should be removed.

    def push(self, value: str):
        message: Dict = {
            "chunk": value,
            "expiration": (datetime.now() + timedelta(seconds=self.TTL_in_seconds)),
        }

        self.prompt_accumulator.append(message)

    def reset(self):
        self.prompt_accumulator = []

    def refresh_accumulator(self):
        """remove expired prompt chunks"""
        self.prompt_accumulator = [
            prompt
            for prompt in self.prompt_accumulator
            if prompt["expiration"] > datetime.now()
        ]

    @property
    def prompt(self):
        """Return non-expired prompt chunks"""
        self.refresh_accumulator()
        return " ".join(
            [prompt_record["chunk"] for prompt_record in self.prompt_accumulator]
        )

    @property
    def isRepeat(self):
        """returns true if the user says the samething twice. TODO do semantic matching."""
        if len(self.prompt_accumulator) < 2:
            return False
        else:
            return self.prompt_accumulator[-1] == self.prompt_accumulator[-2]
