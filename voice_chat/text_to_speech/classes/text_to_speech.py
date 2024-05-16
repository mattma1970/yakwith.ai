from abc import ABC, abstractmethod
from typing import Any, Union, Iterable, List, Generator, IO, Tuple, Optional, Dict
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
        """Remove unvoicable characters and text splitting for lantency optimization."""
        pass

    @abstractmethod
    def audio_stream_generator(self, text: str) -> Tuple[bytes, Optional[Dict]]:
        """Create a generator that returns audio byte data."""
        pass

    @abstractmethod
    def audio_viseme_generator(
        self, text: str, overlap: str = ""
    ) -> Tuple[bytes, List, List]:
        """Create the generator that returns audio chunks and the aligned visemes and blendshapes."""
        pass
