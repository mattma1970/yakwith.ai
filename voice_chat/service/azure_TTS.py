""" This module is for text to speech from Azure. 
    Note: the azure-cognitiveservices-speech sdk in ubuntu requires openssl 1.x and does not work for the default v3.0 in ubunt 22.04
    See installation instructions here. 
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/quickstarts/setup-platform?pivots=programming-language-python&tabs=linux%2Cubuntu%2Cdotnetcli%2Cdotnet%2Cjre%2Cmaven%2Cnodejs%2Cmac%2Cpypi

"""

import os
import azure.cognitiveservices.speech as speechsdk
from attrs import define, field
from typing import List, Any, Dict, Generator, Iterable
from griptape.artifacts import TextArtifact

import re

MIN_LENGTH_FOR_SYNTHESIS = 5



@define
class AzureTextToSpeech:
    voice_id: str = field(default="en-US-JasonNeural", kw_only=True)
    tts_sentence_end: List = field(factory=list, kw_only=True)

    speech_synthesizer: speechsdk.SpeechSynthesizer = field(init=False)
    full_message: str = field(init=False)

    def __attrs_post_init__(self):
        # tts sentence end mark used to find natural breaks for chunking data to send to TTS
        self.tts_sentence_end = [".", "!", "?", ";", "。", "！", "？", "；", "\n"]
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_SERVICES_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),           
        )
        speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "/home/mtman/Documents/Repos/yakwith.ai/voice_chat/logs/TTS/log.log")

        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config, audio_config
        )  # TODO for testing only need to stream it back.
        self.full_message = ""

    def audio_generator(self, text_stream: Iterable[TextArtifact]):
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
                        print(
                            f"Speech synthesized partial text of: {text_for_synth}"
                        )
                        yield text_for_synth
                        text_for_synth = text_chunk[split:]
                else:
                    text_for_synth += text_chunk
            else:
                text_for_synth += text_chunk
        """ Ensure somethgn is always returned in case text_stream isn't terminated by an end of sentance marker."""
        if text_for_synth != "":
            return text_for_synth
