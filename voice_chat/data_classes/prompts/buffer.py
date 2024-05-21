"""
Class to manage the accumulation of prompt fragments until a sensible whole is found
"""

from typing import Dict
from attr import define, field
from datetime import datetime
from datetime import timedelta


@define
class PromptBuffer:
    prompt_buffer: list[Dict] = field(factory=list)
    TTL_in_seconds: int = field(
        default=20
    )  # number of seconds after which a chunk of prompt should be removed.

    def __len__(self):
        """
        Return the number of valid prompt fragments in the buffer
        """
        self.refresh_accumulator()
        return len(self.prompt_buffer)

    def _normalize_chunk(self, text: str):
        """
        Text pushed onto the buffer should be cleaned up on ingestion."""
        ret: str = text.lstrip(" ")
        ret = ret[0].lower() + ret[1:].rstrip(",.!? ")
        return ret

    def push(self, value: str):
        message: Dict = {
            "chunk": self._normalize_chunk(value),
            "expiration": (datetime.now() + timedelta(seconds=self.TTL_in_seconds)),
        }
        self.prompt_buffer.append(message)

    def reset(self):
        self.prompt_buffer = []

    def refresh_accumulator(self):
        """remove expired prompt chunks. Capitalize first chunks beginning."""
        self.prompt_buffer = [
            prompt
            for prompt in self.prompt_buffer
            if prompt["expiration"] > datetime.now()
        ]
        if len(self.prompt_buffer) > 0:
            self.prompt_buffer[0]["chunk"] = (
                self.prompt_buffer[0]["chunk"][0].upper()
                + self.prompt_buffer[0]["chunk"][1:]
            )

    @property
    def prompt(self):
        """Return non-expired prompt chunks"""
        self.refresh_accumulator()
        ret = " ".join([prompt_record["chunk"] for prompt_record in self.prompt_buffer])
        return ret + "."

    @property
    def isRepeat(self):
        """returns true if the user says the samething twice. TODO do semantic matching."""
        if len(self.prompt_buffer) < 2:
            return False
        else:
            return self.prompt_buffer[-1] == self.prompt_buffer[-2]
