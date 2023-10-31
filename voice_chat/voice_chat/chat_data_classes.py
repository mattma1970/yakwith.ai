from pydantic import BaseModel
from dataclasses import dataclass
from typing import List
import json
from attrs import define, field


class InferencePrompt(BaseModel):
    role: str
    content: str

class InferenceDialog(BaseModel):
    """ A list of conversations."""
    dialogs: List[List[InferencePrompt]]

class InferenceSessionPrompt(BaseModel):
    """A dialog or turn labelled with the clients session_id"""
    prompt: List[InferencePrompt]
    session_id: str
    
    @classmethod
    def parse(cls, input: str):
        try:
            ret = InferenceSessionPrompt(**json.loads(input))
        except:
            return None
        return ret