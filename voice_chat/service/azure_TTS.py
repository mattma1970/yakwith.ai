''' This module is for text to speech from Azure. '''

import os
import azure.cognitiveservices.speech as speechsdk
from attrs import define, field
from typing import List, Any, Dict, Generator
from griptape.artifacts import TextArtifact

import re

@define
class AzureTextToSpeech:

    voice_id:str = field( default = 'en-US-JasonNeural', kw_only=True )
    tts_sentence_end : List = field (factory = list, kw_only=True)

    speech_synthesizer: speechsdk.SpeechSynthesizer = field (init = False)
    collected_messages: List = field(init=False)


    def __attrs_post_init__(self):

        # tts sentence end mark used to find natural breaks for chunking data to send to TTS
        self.tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        speech_config = speechsdk.SpeechConfig(
                        subscription=os.getenv("AZURE_SPEECH_SERVICES_KEY"), 
                        region=os.getenv("AZURE_SPEECH_REGION")
                    )
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)  # TODO for testing only need to stream it back. 
        self.collected_messages = []

    def audio_generator(self, text_stream):
        for chunk in text_stream:
            if len(chunk) > 0:
                sentance_ends_flags = list(map(lambda x: x in self.tts_sentance_end))
                if any(sentance_ends_flags):
                    # split on the last end of sentance marker
                    split = len(sentance_ends_flags) - sentance_ends_flags.reverse().index(True)
                    self.collected_messages.append(chunk[:split])
                    text_for_synthesis = ''.join(self.collected_messages).strip()
                    collected_messages = [chunk[split:]]
                    if text_for_synthesis != '': # if sentence only have \n or space, we could skip
                        print(f"Speech synthesized partial text of: {text_for_synthesis}")
                        yield text_for_synthesis
                else:
                    self.collected_messages.append(chunk) 
