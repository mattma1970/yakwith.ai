# Try this

from typing import Tuple, Dict, List, Generator, Any, Optional, IO, Iterable, Union
import re
import requests, json, base64
from datetime import timedelta
from attrs import define, Factory, field
from voice_chat.text_to_speech.classes.text_to_speech import TextToSpeechClass
from dotenv import load_dotenv
import os
from io import BytesIO
import statistics

from griptape.artifacts import TextArtifact
from voice_chat.utils.text_processing import remove_problem_chars, remove_strings
from voice_chat.utils.tts_utilites import TTSUtilities
from voice_chat.text_to_speech.classes.audio_response import AudioResponse

from voice_chat.lipsync.LipsyncEn import LipsyncEn
from collections import defaultdict

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
    viseme_stats_raw: defaultdict = defaultdict(list)

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

    def phenome_trainer(
        self, text: Union[str, List], viseme_processor: LipsyncEn
    ) -> List[Dict]:
        """Gather statistics on the duration of the visemes generated.
        The Elevenlabs API can return character level timestamps. These are used to determine the duration of the visemes relative normalised to the number of visemes in the string.

        @args:
            text: the text to be converted
            viseme_processor: Any: a class that has text to viseme mapping function. e.g. Lipsync
        @returns:
            key,value pairs of viseme and normalized durations where a duration of 1 is the average time in seconds of a viseme returned by the string.
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        for _text in texts:
            processed_text = viseme_processor.pre_process_text(_text)
            _, timestamp_data = self.audio_stream_generator(
                _text
            )  # get just the timestamp data, a dictionary with characters, startime and end time of the character.
            viseme_dict = viseme_processor.viseme_segmenter(
                processed_text
            )  # {words, viseme_mapping[(words, start_char, end_char, number of visemes in text chunk)]
            # ground_truth = viseme_processor.words_to_visemes(processed_text)
            # Line up the character level data with the visemes
            average_duration_seconds: float = (
                timestamp_data["character_end_times_seconds"][-1]
                - timestamp_data["character_start_times_seconds"][0]
            ) / len(viseme_dict["viseme_mapping"])
            for mapping in viseme_dict["viseme_mapping"]:
                _phenome_duration = (
                    (
                        timestamp_data["character_end_times_seconds"][mapping[2] - 1]
                        - timestamp_data["character_start_times_seconds"][mapping[1]]
                    )
                    / float(mapping[-1])
                    / average_duration_seconds
                )
                self.viseme_stats_raw[mapping[0]].append(_phenome_duration)

        ret: Dict = {}
        for k, v in self.viseme_stats_raw.items():
            ret[k] = statistics.mean(v)
        return ret


if __name__ == "__main__":
    vs_proc = LipsyncEn()
    elevenlabs = ElevenLabsTextToSpeech()
    long_text = """
    Hello, everyone! This is the LONGEST TEXT EVER! I was inspired by the various other "longest texts ever" on the internet, and I wanted to make my own. So here it is! This is going to be a WORLD RECORD! This is actually my third attempt at doing this. The first time, I didn't save it. The second time, the Neocities editor crashed. Now I'm writing this in Notepad, then copying it into the Neocities editor instead of typing it directly in the Neocities editor to avoid crashing. It sucks that my past two attempts are gone now. Those actually got pretty long. Not the longest, but still pretty long. I hope this one won't get lost somehow. Anyways, let's talk about WAFFLES! I like waffles. Waffles are cool. Waffles is a funny word. There's a Teen Titans Go episode called "Waffles" where the word "Waffles" is said a hundred-something times. It's pretty annoying. There's also a Teen Titans Go episode about Pig Latin. Don't know what Pig Latin is? It's a language where you take all the consonants before the first vowel, move them to the end, and add '-ay' to the end. If the word begins with a vowel, you just add '-way' to the end. For example, "Waffles" becomes "Afflesway". I've been speaking Pig Latin fluently since the fourth grade, so it surprised me when I saw the episode for the first time. I speak Pig Latin with my sister sometimes. It's pretty fun. I like speaking it in public so that everyone around us gets confused. That's never actually happened before, but if it ever does, 'twill be pretty funny. By the way, "'twill" is a word I invented recently, and it's a contraction of "it will". I really hope it gains popularity in the near future, because "'twill" is WAY more fun than saying "it'll". "It'll" is too boring. Nobody likes boring. This is nowhere near being the longest text ever, but eventually it will be! I might still be writing this a decade later, who knows? But right now, it's not very long. But I'll just keep writing until it is the longest! Have you ever heard the song "Dau Dau" by Awesome Scampis? It's an amazing song. Look it up on YouTube! I play that song all the time around my sister! It drives her crazy, and I love it. Another way I like driving my sister crazy is by speaking my own made up language to her. She hates the languages I make! The only language that we both speak besides English is Pig Latin. I think you already knew that. Whatever. I think I'm gonna go for now. Bye! Hi, I'm back now. I'm gonna contribute more to this soon-to-be giant wall of text. I just realised I have a giant stuffed frog on my bed. I forgot his name. I'm pretty sure it was something stupid though. I think it was "FROG" in Morse Code or something. Morse Code is cool. I know a bit of it, but I'm not very good at it. I'm also not very good at French. I barely know anything in French, and my pronunciation probably sucks. But I'm learning it, at least. I'm also learning Esperanto. It's this language that was made up by some guy a long time ago to be the "universal language". A lot of people speak it. I am such a language nerd. Half of this text is probably gonna be about languages. But hey, as long as it's long! Ha, get it? As LONG as it's LONG? I'm so funny, right? No, I'm not. I should probably get some sleep. Goodnight! Hello, I'm back again. I basically have only two interests nowadays: languages and furries. What? Oh, sorry, I thought you knew I was a furry. Haha, oops. Anyway, yeah, I'm a furry, but since I'm a young furry, I can't really do as much as I would like to do in the fandom. When I'm older, I would like to have a fursuit, go to furry conventions, all that stuff. But for now I can only dream of that. Sorry you had to deal with me talking about furries, but I'm honestly very desperate for this to be the longest text ever. Last night I was watching nothing but fursuit unboxings. I think I need help. This one time, me and my mom were going to go to a furry Christmas party, but we didn't end up going because of the fact that there was alcohol on the premises, and that she didn't wanna have to be a mom dragging her son through a crowd of furries. Both of those reasons were understandable. Okay, hopefully I won't have to talk about furries anymore. I don't care if you're a furry reading this right now, I just don't wanna have to torture everyone else. I will no longer say the F word throughout the rest of this entire text. Of course, by the F word, I mean the one that I just used six times, not the one that you're probably thinking of which I have not used throughout this entire text. I just realised that next year will be 2020. That's crazy! It just feels so futuristic! It's also crazy that the 2010s decade is almost over. That decade brought be a lot of memories. In fact, it brought be almost all of my memories. It'll be sad to see it go. I'm gonna work on a series of video lessons for Toki Pona. I'll expain what Toki Pona is after I come back. Bye! I'm back now, and I decided not to do it on Toki Pona, since many other people have done Toki Pona video lessons already. I decided to do it on Viesa, my English code. Now, I shall explain what Toki Pona is. Toki Pona is a minimalist constructed language that has only ~120 words! That means you can learn it very quickly. I reccomend you learn it! It's pretty fun and easy! Anyway, yeah, I might finish my video about Viesa later. But for now, I'm gonna add more to this giant wall of text, because I want it to be the longest! It would be pretty cool to have a world record for the longest text ever. Not sure how famous I'll get from it, but it'll be cool nonetheless. Nonetheless. That's an interesting word. It's a combination of three entire words. That's pretty neat. Also, remember when I said that I said the F word six times throughout this text? I actually messed up there. I actually said it ten times (including the plural form). I'm such a liar! I struggled to spell the word "liar" there. I tried spelling it "lyer", then "lier". Then I remembered that it's "liar". At least I'm better at spelling than my sister. She's younger than me, so I guess it's understandable. "Understandable" is a pretty long word. Hey, I wonder what the most common word I've used so far in this text is. I checked, and appearantly it's "I", with 59 uses! The word "I" makes up 5% of the words this text! I would've thought "the" would be the most common, but "the" is only the second most used word, with 43 uses. "It" is the third most common, followed by "a" and "to". Congrats to those five words! If you're wondering what the least common word is, well, it's actually a tie between a bunch of words that are only used once, and I don't wanna have to list them all here. Remember when I talked about waffles near the beginning of this text? Well, I just put some waffles in the toaster, and I got reminded of the very beginnings of this longest text ever. Okay, that was literally yesterday, but I don't care. You can't see me right now, but I'm typing with my nose! Okay, I was not able to type the exclamation point with just my nose. I had to use my finger. But still, I typed all of that sentence with my nose! I'm not typing with my nose right now, because it takes too long, and I wanna get this text as long as possible quickly. I'm gonna take a break for now! Bye! Hi, I'm back again. My sister is beside me, watching me write in this endless wall of text. My sister has a new thing where she just says the word "poop" nonstop. I don't really like it. She also eats her own boogers. I'm not joking. She's gross like that. Also, remember when I said I put waffles in the toaster? Well, I forgot about those and I only ate them just now. Now my sister is just saying random numbers. Now she's saying that they're not random, they're the numbers being displayed on the microwave. Still, I don't know why she's doing that. Now she's making annoying clicking noises. Now she's saying that she's gonna watch Friends on three different devices. Why!?!?! Hi its me his sister. I'd like to say that all of that is not true. Max wants to make his own video but i wont let him because i need my phone for my alarm.POOP POOP POOP POOP LOL IM FUNNY. kjnbhhisdnhidfhdfhjsdjksdnjhdfhdfghdfghdfbhdfbcbhnidjsduhchyduhyduhdhcduhduhdcdhcdhjdnjdnhjsdjxnj Hey, I'm back. Sorry about my sister. I had to seize control of the LTE from her because she was doing keymash. Keymash is just effortless. She just went back to school. She comes home from school for her lunch break. I think I'm gonna go again. Bye! Hello, I'm back. Let's compare LTE's. This one is only 8593 characters long so far. Kenneth Iman's LTE is 21425 characters long. The Flaming-Chicken LTE (the original) is a whopping 203941 characters long! I think I'll be able to surpass Kenneth Iman's not long from now. But my goal is to surpass the Flaming-Chicken LTE. Actually, I just figured out that there's an LTE longer than the Flaming-Chicken LTE. It's Hermnerps LTE, which is only slightly longer than the Flaming-Chicken LTE, at 230634 characters. My goal is to surpass THAT. Then I'll be the world record holder, I think. But I'll still be writing this even after I achieve the world record, of course. One time, I printed an entire copy of the Bee Movie script for no reason.It'll feel nice to be way ahead the record. My sister's alarm clock has been going off for half an hour and I haven't turned it off. Why? Because LAZYNESS! Actually, I really should turn it off now. There, I turned it off. First when I tried to turn it off, it started playing the radio. Then I tried again, and it turned off completely. Then I hurt myself on the door while walking out. So that was quite the adventure. I'm gonna go sleep now. Goodnight! Hey, I'm back again. My computer BSOD'd while writing this, so I have to start this section over again. 
    That's why you save your work, kids! Before I had to start over again, I was talking about languages. 
    Yes, I decided to bring that topic back after a while. But I no longer want to talk about it. 
    Why? Because it'll probably bore you to death. That is assuming you're reading this at all. 
    Who knows, maybe absolutely zero people will read this within the span of the universe's existence. 
    But I doubt that. There's gotta be someone who'll find this text and dedicate their time to reading it, even if it takes thousands of years for that to happen. 
    What will happen to this LTE in a thousand years? Will the entire internet dissapear within that time? In that case, will this text dissapear with it? 
    Or will it, along with the rest of what used to be the internet, be preserved somewhere? I'm thinking out loud right now. Well, not really "out loud"
      because I'm typing this, and you can't technically be loud through text. THE CLOSEST THING IS TYPING IN ALL CAPS. Imagine if I typed this entire text like that. 
      That would be painful. I decided to actually save my work this time, in case of another crash. I already had my two past attempts at an LTE vanish from existance. I mean, most of this 
      LTE is already stored on Neocities, so I probably won't need to worry about anything. I think I might change the LTE page a little. I want the actual text area to be larger. 
      I'm gonna make it a very basic HTML page with just a header and text. Maybe with some CSS coloring. I don't know. Screw it, I'm gonna do it. There, now the text area is larger.
        It really does show how small this LTE is so far compared to FlamingChicken or Hermnerps. But at least I made the background a nice Alice Blue. That's the name of the CSS color I used. 
    """
    text_array = re.split(r"\.", long_text)
    text_array = list(map(lambda x: x.strip(), text_array))
    text_array = text_array[:15]

    elevenlabs.phenome_trainer(text_array, vs_proc)
