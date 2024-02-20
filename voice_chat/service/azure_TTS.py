""" This module is for text to speech from Azure. 
    Note: the azure-cognitiveservices-speech sdk in ubuntu requires openssl 1.x and does not work for the default v3.0 in ubunt 22.04
    See installation instructions here. 
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/quickstarts/setup-platform?pivots=programming-language-python&tabs=linux%2Cubuntu%2Cdotnetcli%2Cdotnet%2Cjre%2Cmaven%2Cnodejs%2Cmac%2Cpypi
    voices
    https://learn.microsoft.com/en-gb/azure/ai-services/speech-service/language-support?tabs=stt#text-to-speech

"""

import os
import math
import azure.cognitiveservices.speech as speechsdk
from attrs import define, field, Factory
from typing import List, Any, Dict, Generator, Iterable, Callable, Tuple
from griptape.artifacts import TextArtifact
from voice_chat.utils.text_processing import remove_problem_chars
from voice_chat.utils.tts_utilites import TTSUtilities
from utils import TimerContextManager, createIfMissing

import re, json
import logging

logger = logging.getLogger("YakChatAPI")


@define
class AzureTextToSpeech:
    voice_id: str = field(
        default="en-AU-KimNeural", kw_only=False
    )  # Ensure this is a valid voice id from you TTS provider
    audio_config: speechsdk.audio.AudioOutputConfig = field(
        default=speechsdk.audio.AudioOutputConfig(use_default_speaker=True),
        kw_only=True,
    )  # can be overwritten

    speech_synthesizer: speechsdk.SpeechSynthesizer = field(init=False)
    full_message: str = field(init=False)
    blendshape_options: dict = field(factory=dict)

    def __attrs_post_init__(self):

        self.blendshape_options = {
            "visemes_only": "redlips_front",
            "blendshapes": "FacialExpression",
        }

        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_SERVICES_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )  # audio/mpeg
        speech_config.speech_synthesis_voice_name = self.voice_id

        # Log SDK output
        if os.environ["AZURE_ENABLE_SPEECH_SDK_LOGGING"]:
            log_path = (
                f"{os.environ['APPLICATION_ROOT_FOLDER']}/voice_chat/logs/TTS/log.log"
            )
            createIfMissing(log_path)
            speech_config.set_property(
                speechsdk.PropertyId.Speech_LogFilename,
                log_path,
            )

        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config, self.audio_config
        )  # TODO for testing only need to stream it back.
        self.full_message = ""

        logger.debug(f"Created Azure Speech Synthesizer.")

    def text_preprocessor(
        self,
        text_stream: Iterable[TextArtifact],
        filter: str = None,
        use_ssml: bool = True,
    ):
        """
        Generator of text chunks to be sent for speech synthesis.
        Accumulates the streaming text and yeilds at natural boundaries in the text.
        For latency improvement, the first chunck sent for synthesis should be as short as possible.
        Short chunks, less than a sentance long, are synthesized as if they were complete sentances and so have
        incorrect, falling, intonation at the end. The solution to create overlap between chunks for the first text chunk.

        Arguments:
            text_stream. TextArtificat generator
            filter: str: A valid regex that passes acceptable characters (useful for removing punctuation)
        Yields:
            Tuple[str,str]: Text to be generated and cached, additional words used for correcting intonation of short, sub-sentance phrases.
        """

        text_accumulator = ""
        text_for_accumulation = ""
        is_first_sentance: bool = (
            True  # first chunk of response yeilded needs to be optimised for speed.
        )

        for chunk in text_stream:
            # Acculate the text until a natural break in text is found. Reduce the accumulator as text is sent for synthesis.
            text_accumulator += chunk.value
            self.full_message += chunk.value  # For caching etc.

            min_length = (
                int(os.environ["MAX_CHARACTERS_FOR_SUBSEQUENT_SYNTHESIS"])
                if is_first_sentance is False
                else int(os.environ["WORD_COUNT_FOR_FIRST_SYNTHESIS"])
                * 10  # TODO hack. remove
            )
            if len(chunk.value) > 0 and len(text_accumulator) >= min_length:

                phrase, overlap, remainder = "", "", ""

                if is_first_sentance:
                    phrase, overlap, remainder = TTSUtilities.get_first_utterance(
                        text_accumulator,
                        int(os.environ["WORD_COUNT_FOR_FIRST_SYNTHESIS"]),
                        int(os.environ["WORD_COUNT_OVERLAP_FOR_FIRST_SYNTHESIS"]),
                    )
                    is_first_sentance = False
                else:
                    phrase_end_index: int = min_length
                    for sentence_marker in self.tts_sentence_end_regex:
                        # Else be greedy with the text size.
                        for match in re.finditer(sentence_marker, text_accumulator):
                            # Get the last natural break position over all the sentance markers
                            if match.start() > phrase_end_index:
                                phrase_end_index = match.start()
                        phrase, overlap, remainder = (
                            text_accumulator[:phrase_end_index],
                            "",
                            text_accumulator[phrase_end_index:],
                        )

                if (
                    phrase.strip() != ""
                ):  # if sentence only have \n or space, we could skip
                    preprocessed_phrase = self.prepare_for_synthesis(
                        filter, use_ssml, phrase
                    )
                    yield preprocessed_phrase, overlap
                    text_accumulator = remainder.lstrip()  # Keep the remaining text

        if text_accumulator != "":
            logger.debug(f"Text for synth flushed:{text_accumulator}")
            preprocessed_phrase = self.prepare_for_synthesis(
                filter, use_ssml, text_accumulator
            )
            yield preprocessed_phrase, ""

    def send_audio_to_speaker(self, text: str) -> None:
        """send to local speaker on server as per audio configuration."""
        if self.audio_config == None:
            raise RuntimeWarning(
                "Speech synthesizer is not configured for using local speaker. No audio will be played."
            )
        self.speech_synthesizer.speak_text_async(text).get()

    def audio_stream_generator(self, text: str) -> speechsdk.SpeechSynthesisResult:
        """
        Generate speech using ssml. No visemes or blendshapes.
        @args:
            text: str: plain text to generate

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
        ssml: str = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
                            <voice name="{self.voice_id}">
                                 {text}
                        </voice>
                    </speak>"""
        result = self.speech_synthesizer.speak_ssml_async(ssml).get()
        return result


@define
class AzureTTSViseme(AzureTextToSpeech):
    """
    Extends the streaming speech synthesizer to also include viseme and blendshape data.
    While visemes are always returned, blendshapes are optional. If blendshapes are not generated, the blendshap_log
    will still be returned but will be empty.
    """

    viseme_log: List[Dict] = field(factory=list)
    index: int = 0
    viseme_callback: Callable = field(init=False)
    blendshape_type: str = field(
        default="blendshapes"
    )  # options visemes_only, blendshapes(<= both visemes and facial blendshapes)
    use_blendshapes: bool = field(default=True)
    blendshapes_log: List[str] = field(factory=list)
    wordboundary_log: List[float] = field(factory=list)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.viseme_callback = self.create_viseme_cb()

        self.speech_synthesizer.viseme_received.connect(
            self.viseme_callback
        )  # Subscribe the speech synthesizer to the viseme events

        self.subscribe_to_wb_events()

    # Callback section

    # Word boundary callbacks (used by short utterance functions)
    def create_wb_cb(self) -> Callable:
        def _wb_cb(evt):
            logger.debug(f"Word boundary: {(evt.audio_offset+5000)/10000} {evt.text}")
            self.wordboundary_log.append((evt.audio_offset + 5000) / 10000)

        return _wb_cb

    def unsubscribe_to_wb_events(self):
        self.speech_synthesizer.synthesis_word_boundary.disconnect_all()

    def subscribe_to_wb_events(self):
        self.speech_synthesizer.synthesis_word_boundary.connect(self.create_wb_cb())

    def clear_wordboundary_log(self):
        self.wordboundary_log = []

    def create_viseme_cb(self) -> Callable:
        def _viseme_logger(evt):
            """Call back to capture viseme and blendshapes"""
            start: float = evt.audio_offset / 10000000  # Time in seconds
            msg: Dict = {"start": start, "end": 10000.0, "value": evt.viseme_id}

            if evt.animation == "":
                # logger.debug(evt)
                if self.index > 0:
                    self.viseme_log[-1][
                        "end"
                    ] = start  # Update the end time marker as the start of the current time.
                self.viseme_log.append(msg)
                # logger.debug(f'index: {self.index},{msg}')
                self.index += 1
            else:
                # evt.animation will be empty string if only visemes are generated.
                animation: str = json.loads(evt.animation)
                # logger.debug(f'{evt} + Animation FrameIndex: {animation["FrameIndex"]}')
                self.blendshapes_log.extend(
                    animation["BlendShapes"]
                )  # Drop frameindex for speed of parsing at client.
            return None

        return _viseme_logger

    def audio_viseme_generator(self, text: str):
        """
        Return a tuple of an audio snippet and viseme log containing the time stamps of the viseme events.
        Note: visemes and blendshapes generated asyn via the viseme_cb callback invokations and pushed to the respective logs.
        """
        self.clear_wordboundary_log()  # Reset wb log before evary chunk is generated.
        with TimerContextManager(f"SpeechSynth:{text}", logger, logging.DEBUG) as timer:
            audio_output: speechsdk.SpeechSynthesisResult = self.audio_stream_generator(
                text
            )
        if audio_output.cancellation_details is not None:
            logger.debug(
                f"Reason: {audio_output.reason}, details: {audio_output.cancellation_details.error_details}"
            )
        if audio_output.audio_duration:  # Only available when synthesis is complete
            logger.debug(f"Audio Duration: {audio_output.audio_duration} (s)")
        _viseme_log = self.viseme_log.copy()
        _blendshape_log = self.blendshapes_log.copy()

        self.reset_viseme_log(len(_viseme_log), len(_blendshape_log))
        return audio_output, _viseme_log, _blendshape_log

    def audio_stream_generator(self, text: str) -> speechsdk.SpeechSynthesisResult:
        """
        Generate speech. Visemes are always returned. Blendshapes, that will depend on whether use_blendshapes is set.
        @args:
            text: str: plain text to generate
            use_blendshapes: bool: indicates if blendshapes should be retrieved from Azure.

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
        if self.use_blendshapes:
            self.blendshape_type = "blendshapes"
        else:
            self.blendshape_type = "visemes_only"

        # ssml provides greater flexibility to control generation e.g. pitch, rate, intonation etc.
        ssml: str = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
                            <voice name="{self.voice_id}">
                                <mstts:viseme type="{ self.blendshape_options[self.blendshape_type]}"/>
                                <mstts:silence  type="tailing-exact" value="0ms"/>
                                <mstts:silence type="leading-exact" value="0ms"/>
                                <mstts:silence type="sentenceboundart-exact" value="0ms"/>
                                 {text}
                        </voice>
                    </speak>"""
        result = self.speech_synthesizer.speak_ssml_async(ssml).get()
        return result

    def reset_viseme_log(self, start_index: int, blendshape_start) -> None:
        """
        Its possible that move events have occured since the audio stream yeilded chunks fo the viseme_log
        should be truncated to remove the old visemes.
        """
        if start_index > 0:
            self.index -= start_index
            self.viseme_log = self.viseme_log[: self.index]
            self.blendshapes_log = self.blendshapes_log[:blendshape_start]
            # logger.debug(f"reset viseme log : {self.index}")
        else:
            self.viseme_log = []
            self.blendshapes_log = []
