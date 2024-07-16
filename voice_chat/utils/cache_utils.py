import pickle
import os
import re
from data_classes.cache import CacheHelper, QueryType
from yak_agents import YakAgent
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class CacheUtils:
    def __init__(self):
        pass

    @classmethod
    def cache_if_short_utterance(
        cls,
        prompt: str,
        cache: CacheHelper,
        yak_agent: YakAgent,
        audio_data: bytes,
        visemes: List[Dict],
        blendshapes: List = [],
    ) -> None:
        """
        Cache short text chunks. Used for latency optimization.
        They short chunks are the output of the TTS so do not have to be 'pronoun-aware' as do the input prompts.
        @args:
            prompt: the text from which the first utterance is pulled.
            cache: the cache. e.g redis.
            yak_agent: Griptape agent
            TTS: The text to speech object.
            audio_data, visemes, blendshapes: string representatiosn of the relevant data.
        """
        prompt_array = re.split(r"\s+", prompt.lstrip())

        # We can only cache the data if the prompt is a completed stream.
        #    It must be a completed stream or otherwise the viseme and (optional) blendshape data won't be available.

        if (
            len(prompt_array) > int(os.environ["WORD_COUNT_FOR_FIRST_SYNTHESIS"])
            or audio_data == b""
            or len(visemes) == 0
        ):
            return

        cache_response_key: str = cache.get_cache_key(
            QueryType.RESPONSE, yak_agent.voice_id, prompt
        )
        if cache.exists(cache_response_key) is False:
            cache.hset(cache_response_key, "response", prompt)
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
        id: str = "",
    ):
        """
        Hit the cache.
        @args:
            type: str : ["res","req"] response or request.
            id: str: optional tag used to identify the audio. For example a MenuID for the menu from which the response is taken.
        @returns:
            response: str: default ""
            visemes, blendshapes, audio from cache.
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
            except Exception as e:
                logger.error(e)
                pass
        return response, visemes, blendshapes, audio
