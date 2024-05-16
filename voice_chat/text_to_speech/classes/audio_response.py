from attrs import define, field, Factory
from datetime import timedelta


@define
class AudioResponse:
    audio_data: bytes = field(default=None)
    audio_duration: timedelta = field(default=None)
