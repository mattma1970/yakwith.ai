import pickle, os
from voice_chat.data_classes.cache import CacheHelper, QueryType
from voice_chat.yak_agents import YakAgent
from typing import Any, List, Dict
from enum import Enum


class CacheUtils:
    def __init__(self):
        pass

    @classmethod
    def cache_if_short_utterance(
        cls,
        prompt: str,
        cache: CacheHelper,
        yak_agent: YakAgent,
        audio_data: str,
        visemes: List[Dict],
        blendshapes: List = [],
    ):
        """
        Cache short text chunks. Used for latency optimization.
        These utterances are not related to the request or the conversation context so
        @args:
            prompt: the text from which the first utterance is pulled.
            cache: the cache. e.g redis.
            yak_agent: Griptape agent
            TTS: The text to speech object.
            audio_data, visemes, blendshapes: string representatiosn of the relevant data.
        """
        phrase: str = ""
        if (
            len(prompt) <= int(os.environ["MIN_LENGTH_FOR_FIRST_SYNTHESIS"])
            and audio_data
            and visemes
        ):
            """We can only cache the data if the prompt is a completed stream.
            It must be a completed stream or otherwise the viseme and (optional) blendshape data won't be available.
            """
            phrase = prompt
        else:
            return

        cache_response_key: str = cache.get_cache_key(
            QueryType.RESPONSE, yak_agent.voice_id, phrase
        )
        if cache.exists(cache_response_key) == False:
            cache.hset(cache_response_key, "response", phrase)
            cache.append_to_cache(cache_response_key, "blendshapes", blendshapes)
            cache.append_to_cache(cache_response_key, "visemes", visemes)
            cache.append_to_cache(cache_response_key, "audio", audio_data)

    @classmethod
    def get_from_cache(
        cls,
        type: QueryType,
        phrase: str,
        yak: YakAgent,
        cache: CacheHelper,
        *,
        id: str = ""
    ):
        """
        Hit the cache.
        @args: type: str : ["res","req"] response or request.
        @returns:
            response: str: default ""
            other: str|list|None.

        """
        response, visemes, blendshapes, audio = "", None, None, None
        phrase_cache_key = cache.get_cache_key(type, yak.voice_id, phrase, id)
        cached_data = cache.hgetall(phrase_cache_key)
        if len(cached_data) > 0:
            try:
                response, visemes, audio = (
                    cached_data[b"response"],
                    pickle.loads(cached_data[b"visemes"]),
                    pickle.loads(cached_data[b"audio"]),
                )
                if b"blendshapes" in cached_data:
                    # blendshapes is optional
                    blendshapes = pickle.loads(cached_data[b"blendshapes"])
            except:
                pass
        return response, visemes, blendshapes, audio
