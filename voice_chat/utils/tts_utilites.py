"""
    Miscellaneous utilities used to manipulate text and audio for TTS operations that are not unique to a STT provider.
"""

import math
import re
from typing import Dict, List, Any, Union, Tuple
import logging
from pydub import AudioSegment
import io, os
import voice_chat.utils.text_processing as TextProcessing

logger = logging.getLogger(__name__)


class TTSUtilities:

    @classmethod
    def get_sentance_break_regex(cls) -> str:
        return [r"\.(?![0-9])", r"[!?,;:]"]

    def get_sentance_end_regex(cls):
        return [r"\.(?![0-9])", r"[!?;:]"]

    @classmethod
    def total_samples(cls, timestamp_in_ms: int, sample_rate: int = 16000):
        return math.floor(timestamp_in_ms / 1000 * sample_rate)

    @classmethod
    def truncate_audio_byte_data(
        cls,
        audio_bytes: bytes,
        segment_duration: float,
        sample_rate: int = 16000,
        bits_per_sample: int = 16,
        format: str = "mp3",
    ):
        """
        Truncate audio byte data (not base64encoded) to a duration of segment_duration in milliseconds.
        @args:
            audio_bytes: byte: Raw audio . Azure default is PCM 16khz unless AudioStreamFormat is set in the synthesizer configuration.
            segment_duration: desire new length of audio in milliseconds
            format: pcm or mp3.
        @returns:
            truncated_audio_bytes: bytes
        """
        if not (isinstance(audio_bytes, bytes)):
            logger.error("Non-byte data passed for truncating.")
        else:
            logger.debug(f"Turncation audio to {segment_duration} ms")
            if format.strip().lower() == "mp3":
                mp3_file_like = io.BytesIO(audio_bytes)
                decompressed_audio = AudioSegment.from_mp3(mp3_file_like)
                if len(decompressed_audio) > segment_duration:
                    decompressed_audio = decompressed_audio[:segment_duration]
                    output_buffer = io.BytesIO()
                    decompressed_audio.export(output_buffer, format="mp3")
                    return output_buffer.getvalue()
                else:
                    return audio_bytes

            elif format.strip().lower() == "pcm":
                decompressed_audio = audio_bytes
                sample_count: int = cls.total_samples(segment_duration, sample_rate)
                if len(decompressed_audio) > sample_count * (bits_per_sample // 8.0):
                    truncated_audio_bytes = audio_bytes[
                        : (sample_count * bits_per_sample // 8.0)
                    ]
                    return truncated_audio_bytes
                else:
                    return audio_bytes

            else:
                raise RuntimeError("Unsupport audio format set in audio trunction")

    @classmethod
    def get_first_utterance(
        cls,
        text_for_synth: str,
        phrase_length: int = 3,
        overlap_length: int = 0,
    ) -> Tuple[str, str, str]:
        """
            Get segments of text_for_synth for speech generation and short_utterance caching.
            It will return the first phrase_length words unless a natural break in that segment.
        @args:
            text_for_synth: str: an accumulator for the chunks of text returned.
            phrase_length: numbr of words in the phase whose TTS will be played.
            overlap_length: number of additional words beyond the phrase length. Used for correcting intonation 'bug' in sub-sentance audio generation.
        @returns:
            Tuple[str,str,str]: first phrase, next overlap_length words following the first_phrase, all words following the first_phrase
        """
        remainder: str = ""
        overlap: str = ""

        words_for_synth = re.split(r"(?<=\S)\s+(?=\S)", text_for_synth)

        if len(words_for_synth) < phrase_length:
            return phrase, "", ""

        phrase = " ".join(words_for_synth[:phrase_length])
        remainder = " ".join(words_for_synth[phrase_length:])

        for break_marker in cls.get_sentance_break_regex():  # non-space sentance breaks
            match = re.search(break_marker, phrase)
            if match and match.start() > 0:
                return (
                    phrase[: match.start() + 1],
                    "",
                    phrase[(match.start() + 1) :] + " " + remainder,
                )  # include the punctuation mark

        # If we made it here then there isn't natural pause so we'll return the _phrase plus the next overlap_length words.
        try:
            _overlap_word_count = min(
                len(words_for_synth), phrase_length + overlap_length
            )
            overlap = " ".join(words_for_synth[phrase_length:_overlap_word_count])
        except:
            pass

        return phrase, overlap, remainder

    @classmethod
    def prepare_for_synthesis(cls, filter, use_ssml, phrase):
        if filter is not None:
            phrase = TextProcessing.remove_problem_chars(phrase, filter)
        if use_ssml:
            phrase = cls.escape_for_ssml(phrase)
        logger.debug(f"Text for synth: {phrase}")
        return phrase

    @classmethod
    def escape_for_ssml(cls, text: str):
        # Azure ssml doesn't need (or accept " and ' entiry replacements)
        special_chars: dict = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
        }
        for char, replacement in special_chars.items():
            text = text.replace(char, replacement)
        return text
