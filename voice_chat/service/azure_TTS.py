""" This module is for text to speech from Azure. 
    Note: the azure-cognitiveservices-speech sdk in ubuntu requires openssl 1.x and does not work for the default v3.0 in ubunt 22.04
    See installation instructions here. 
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/quickstarts/setup-platform?pivots=programming-language-python&tabs=linux%2Cubuntu%2Cdotnetcli%2Cdotnet%2Cjre%2Cmaven%2Cnodejs%2Cmac%2Cpypi
    voices
    https://learn.microsoft.com/en-gb/azure/ai-services/speech-service/language-support?tabs=stt#text-to-speech

"""

import os
import azure.cognitiveservices.speech as speechsdk
from attrs import define, field
from typing import List, Any, Dict, Generator, Iterable
from griptape.artifacts import TextArtifact
from voice_chat.utils.text_processing import remove_problem_chars

import re

MIN_LENGTH_FOR_SYNTHESIS = 5

import logging

logger = logging.getLogger("YakChatAPI")


@define
class AzureTextToSpeech:
    voice_id: str = field(default="en-US-JasonNeural", kw_only=False)
    audio_config: speechsdk.audio.AudioOutputConfig = field(
        default=speechsdk.audio.AudioOutputConfig(use_default_speaker=True),
        kw_only=True,
    )  # can be overwritten
    tts_sentence_end: List = field(factory=list, kw_only=True)
    speech_synthesizer: speechsdk.SpeechSynthesizer = field(init=False)
    full_message: str = field(init=False)

    def __attrs_post_init__(self):
        # tts sentence end mark used to find natural breaks for chunking data to send to TTS
        self.tts_sentence_end = [".", "!", "?", ";", "。", "！", "？", "；", "\n"]

        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_SERVICES_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )  # audio/mpeg
        speech_config.speech_synthesis_voice_name = "en-AU-KimNeural"

        speech_config.set_property(
            speechsdk.PropertyId.Speech_LogFilename,
            "/home/mtman/Documents/Repos/yakwith.ai/voice_chat/logs/TTS/log.log",
        )

        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config, self.audio_config
        )  # TODO for testing only need to stream it back.
        self.full_message = ""

        logger.debug(f"Creating Azure Speech Synthesizer. Config {speech_config}")

    def text_preprocessor(
        self, text_stream: Iterable[TextArtifact], filter: str = None
    ):
        """
        Accumulates the streaming text and yeilds at natural boundaries in the text.

        Arguments:
            text_stream. TextArtificat generator
            filter: str: A valid regex that passes acceptable characters (useful for removing punctuation)
        Yields:
            text preprocess so that the chunk that is yeilded is split on natural boundaries such sentance end markers or list
        """

        text_for_synth = ""
        for chunk in text_stream:
            text_chunk = chunk.value
            self.full_message += text_chunk
            if len(text_chunk) > 0 and len(text_for_synth) >= MIN_LENGTH_FOR_SYNTHESIS:
                sentance_ends_flags = list(
                    map(lambda x: x in self.tts_sentence_end, text_chunk)
                )
                if any(sentance_ends_flags):
                    sentance_ends_flags.reverse()
                    # split on the last end of sentance marker
                    split = len(sentance_ends_flags) - sentance_ends_flags.index(True)
                    text_for_synth += text_chunk[:split].strip()
                    if (
                        text_for_synth != ""
                    ):  # if sentence only have \n or space, we could skip
                        if filter is not None:
                            text_for_synth = remove_problem_chars(
                                text_for_synth, filter
                            )
                        logger.debug(
                            f"Partial response text for synthesis: {text_for_synth}"
                        )
                        yield text_for_synth
                        text_for_synth = text_chunk[split:]
                else:
                    text_for_synth += text_chunk
            else:
                text_for_synth += text_chunk

        if text_for_synth != "":
            if filter is not None:
                return remove_problem_chars(text_for_synth, filter)
            else:
                return text_for_synth

    def send_audio_to_speaker(self, text: str) -> None:
        """send to local speaker on server as per audio configuration."""
        if self.audio_config == None:
            raise RuntimeWarning(
                "Speech synthesizer is not configured for using local speaker. No audio will be played."
            )
        self.speech_synthesizer.speak_text_async(text).get()

    def audio_stream_generator(self, text: str) -> speechsdk.SpeechSynthesisResult:
        """
        Generate speech for the text passed in.

        Notes:
            https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/samples/python/console/speech_synthesis_sample.py
            Non-blocking but sends entire synthesized speech back rather than streaming back chunks.
            This is acceptable when we are returning one sentance at a time. Otherwise latency will be an issue.

        Returns:
            SpeechSynthesizerResult containing the data for the entire text.
        """
        if self.audio_config != None:
            logger.error(
                f"Speech synthesizer is condfigured for outputing to local speaker and NOT in-memory datastream. audio_config must be set to None."
            )
            raise RuntimeError(
                "Speech synthesizer is condfigured for outputing to local speaker and NOT in-memory datastream."
            )
        result = self.speech_synthesizer.speak_text_async(
            text
        ).get()  # non-blocking but doesn't fullfill promise until all speeech audio is generated.
        return result
