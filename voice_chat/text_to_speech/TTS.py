from abc import ABC, abstractclassmethod, abstractproperty, abstractmethod
from typing import Any, Union, Iterable, List
from enum import Enum


class TextToSpeechClass(ABC):
    @abstractmethod
    def text_preprocessor(
        self,
        text_stream: Iterable,
        filter: str = None,
        use_ssml: bool = True,
        stop_sequences: List[str] = [],
    ):
        pass

    @abstractmethod
    def audio_stream_generator(self, text: str):
        pass
