# Try this

from typing import Tuple, Dict, List, Generator, Any, Optional, IO, Iterable
import re
import requests, json, base64
from datetime import timedelta
from attrs import define, Factory, field
from voice_chat.text_to_speech.classes.text_to_speech import TextToSpeechClass
from dotenv import load_dotenv
import os
from io import BytesIO

from griptape.artifacts import TextArtifact
from voice_chat.utils.text_processing import remove_problem_chars, remove_strings
from voice_chat.utils.tts_utilites import TTSUtilities
from voice_chat.text_to_speech.classes.audio_response import AudioResponse


load_dotenv()
import logging


@define(kw_only=True)
class ElevenLabsTextToSpeech(TextToSpeechClass):
    logger: logging.Logger = field(init=False)
    voice_id: str = field(default="21m00Tcm4TlvDq8ikWAM")
    url: str = field(init=False)
    headers: str = field(init=False)
    permitted_character_regex: str = field(
        default="[^a-zA-Z0-9,. \s'?!;:\$]"
    )  # Azure specific.
    full_message: str = field(default="")

    def __attrs_post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.url = url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/with-timestamps"
        )
        self.headers = {
            "Content-Type": "application/json",
            "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
        }

    def text_preprocessor(
        self,
        text_stream: Iterable[TextArtifact],
        filter: str = None,
        use_ssml: bool = True,
        stop_sequences: List[str] = [],
    ):
        """Pre-processes the agent response text including splitting the first sentance into two segments to reduced time to first audio chunk.
        Args:
            text_stream. TextArtificat generator
            filter: str: A valid regex that passes acceptable characters (useful for removing punctuation)
            stop_sequences: LLMs may not remove the stop sequence strings from the generated text. They must be removed before the regex is run in order to avoid the stop sequence string being corrupted and then sent to TTS
        Yields:
            Tuple[str,str]: Text to be generated and cached, additional words used for correcting intonation of short, sub-sentance phrases.
        """

        text_accumulator = ""
        text_for_accumulation = ""
        is_first_sentance: bool = (
            True  # first chunk of response yeilded needs to be optimised for speed.
        )

        for chunk in text_stream:
            # phrase = chunk.value
            self.logger.debug(chunk)
            phrase: str = remove_strings(chunk.value, stop_sequences)
            phrase = remove_problem_chars(phrase, filter)
            text_accumulator += (
                phrase  # Acculate all text until a natural break in text is found.
            )
            # If chunk has no speakable content the skip (ie. if its only punctuation or spaces etc)
            # logger.info(text_accumulator)
            if bool(re.match(r"^\W+$", phrase)):
                continue

            phrase, overlap, remainder = "", "", ""

            if is_first_sentance:
                if text_accumulator.strip().count(" ") > int(
                    os.environ["WORD_COUNT_FOR_FIRST_SYNTHESIS"]
                ):
                    phrase, overlap, remainder = TTSUtilities.get_first_utterance(
                        text_accumulator,
                        phrase_length=int(os.environ["WORD_COUNT_FOR_FIRST_SYNTHESIS"]),
                        overlap_length=int(
                            os.environ["WORD_COUNT_OVERLAP_FOR_FIRST_SYNTHESIS"]
                        ),
                    )
                    is_first_sentance = False
            else:
                # Else be greedy with the text size.
                phrase_end_index: int = int(
                    os.environ["MIN_CHARACTERS_FOR_SUBSEQUENT_SYNTHESIS"]
                )
                if len(text_accumulator) < phrase_end_index:
                    continue
                # Else be greedy with the text size.
                match_was_found: bool = False
                for sentence_break_regex in TTSUtilities.get_sentance_break_regex():
                    for match in re.finditer(sentence_break_regex, text_accumulator):
                        # Get the last natural break position over all the sentance markers
                        if match and match.start() > phrase_end_index:
                            phrase_end_index = match.start()
                            match_was_found = True

                if match_was_found:
                    phrase, overlap, remainder = (
                        text_accumulator[:phrase_end_index],
                        "",
                        text_accumulator[phrase_end_index:],
                    )
                else:
                    continue

            self.full_message += phrase  # For caching etc.

            if phrase.strip() != "":  # if sentence only have \n or space, we could skip
                preprocessed_phrase = TTSUtilities.prepare_for_synthesis(
                    filter, use_ssml, phrase
                )
                yield preprocessed_phrase, overlap
                text_accumulator = remainder.lstrip()  # Keep the remaining text

        if text_accumulator != "" and not bool(re.match(r"^\W+$", text_accumulator)):
            self.logger.debug(f"Text for synth flushed:{text_accumulator}")
            preprocessed_phrase = TTSUtilities.prepare_for_synthesis(
                filter, use_ssml, text_accumulator
            )
            yield preprocessed_phrase, ""

    def audio_stream_generator(self, text: str) -> Tuple[bytes, Dict]:
        """
        Convert text to audio via the Elevelabs sdk
        Args:
            text: str: text to convert to audio.
        Return:
            Dict[IO[bytes],Dict[]: Audio bytes for entire audio and timestamp data"""

        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        response = requests.post(
            self.url,
            json=data,
            headers=self.headers,
        )

        if response.status_code != 200:
            self.logger.error(
                f"Error encountered, status: {response.status_code}, content: {response.text}"
            )
            quit()

        # response.content contains utf-8 encoded bytes
        json_string = response.content.decode("utf-8")
        # The dict contains 2 keyes: audio_base64 and alignment. Alignment contains character level timing data.
        response_dict = json.loads(json_string)
        audio_data = base64.b64decode(response_dict["audio_base64"])
        timestamp_data: Dict = response_dict["alignment"]

        # TODO - timestamp_data contains character level timing data for alignment with audio.
        return audio_data, timestamp_data

    def audio_viseme_generator(
        self, text: str, overlap: str = ""
    ) -> Tuple[bytes, List, List]:
        """create the generator that returns a tuple of audio. As blendshapes and visemes are not generated the 2nd and 3rd elements of the tuple will be None.
        args:
            text: str: text to convert.
            overlap: str: for some TTS generating addition words and then truncating the audio helps get intonation correct.
        """
        if overlap != "":
            text = text + " " + overlap
        audio_data, timestamp_data = self.audio_stream_generator(text)
        duration: float = timestamp_data["character_end_times_seconds"][-1]
        response: AudioResponse = AudioResponse(audio_data, timedelta(seconds=duration))

        return response, [], []
