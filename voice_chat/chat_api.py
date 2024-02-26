from fastapi import FastAPI, Response, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import requests
import uuid
import shutil
from pathlib import Path
import base64
import math
import urllib
import pickle
import ast
from collections import defaultdict

from PIL import Image
import io

from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any, Iterator, Tuple, Generator
from dotenv import load_dotenv
from uuid import uuid4
from attr import define, field, Factory
import argparse
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import time, datetime
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import azure.cognitiveservices.speech as speechsdk

from voice_chat.yak_agents import YakAgent, YakStatus, ServiceAgent, Task
from voice_chat.data_classes.api import (
    ApiUserMessage,
    AppParameters,
    SessionStart,
    ThirdPartyServiceAgentRequest,
    SttTokenRequest,
    ServiceAgentRequest,
)
from voice_chat.data_classes.chat_data_classes import (
    StdResponse,
    MultiPartResponse,
    BlendShapesMultiPartResponse,
)
from voice_chat.data_classes.data_models import Menu, Cafe, ImageSelector
from voice_chat.data_classes.mongodb import (
    MenuHelper,
    DatabaseConfig,
    ServicesHelper,
    DataHelper,
    ModelChoice,
    ModelHelper,
    ModelChoice,
    ModelHelper,
)
from voice_chat.data_classes.avatar import AvatarConfigParser
from voice_chat.data_classes.redis import RedisHelper
from voice_chat.utils.cache_utils import CacheUtils, QueryType, TTSUtilities

from voice_chat.utils import TimerContextManager

from bson import ObjectId

from voice_chat.utils import DataProxy, createIfMissing, has_pronouns
from voice_chat.text_to_speech.azure_TTS import AzureTextToSpeech, AzureTTSViseme
from voice_chat.text_to_speech.TTS import TextToSpeechClass

from voice_chat.utils import STTUtilities

from griptape.structures import Agent
from griptape.utils import Chat, PromptStack
from griptape.drivers import HuggingFaceInferenceClientPromptDriver
from griptape.events import CompletionChunkEvent, FinishStructureRunEvent
from griptape.rules import Rule, Ruleset
from griptape.utils import Stream
from griptape.artifacts import TextArtifact
from griptape.memory.structure import Run

from omegaconf import OmegaConf, DictConfig

_ALL_TASKS = ["chat_with_agent:post", "chat:post", "llm_params:get"]
_DEFAULT_BUSINESS_UID = "all"
_MENU_RULE_PREFIX = "Below is the menu for a cafe:\n\n"

app = FastAPI()

"""
    Deal with CORS issues of browser calling browser from different ports or names.
    https://fastapi.tiangolo.com/tutorial/cors/
"""
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://app.yakwith.ai",
    "https://dev.d2civivj65v1p0.amplifyapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_registry = {}  # Used to store one agent per session.
service_agent_registry = defaultdict(
    None
)  # For LLM subclasses that do misc tasks like checking for sentance completeness.


def my_gen(response: Iterator[TextArtifact]) -> str:
    for chunk in response:
        yield chunk.value


@app.get("/test_connection")
def test_connection():
    logger.debug("Call to GET test_connection recieved at server")
    return {"data": "ok", "time": datetime.now().strftime("%H:%M:%S")}


@app.post("/get_temp_token")
def get_temp_token(req: SttTokenRequest) -> Dict:
    """
    Get time temporary token api token for requested stt service based on the service_configurations data.
    """
    temp_token = None
    logger.info("Request temporary STT token.")
    # TODO Check that the has a valid client authorization token
    try:
        service_config = DataProxy.get_3p_service_configs(
            authorization_key=req.client_authorization_key,
            authorization_data_source_name="authorized_clients",
            service_name=req.service_name,
            service_data_source_name="service_configs",  # assemblyai_temporary_token
        )
    except Exception as e:
        logger.error(f"Failed to get service token. {req.service_name} {e}")
        raise RuntimeError(f"Failed to get service token. {req.service_name} {e}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": service_config["api_token"],
    }
    data: str = ""
    if "data" in service_config:
        data = json.dumps(
            service_config["data"]
        )  # If data present, it always takes precidence
    else:
        data = json.dumps({"expires_in": service_config["duration"]})

    r = requests.post(
        url=service_config["api_endpoint"],
        headers=headers,
        data=data,
    )

    if r.status_code == 200:
        logger.info("OK: Got temp STT token.")
        response = r.json()
        temp_token = response[service_config["token_key_name"]]
    else:
        logger.error(
            f"Failed to get temp service token for {req.service_name}: Err code {r.status_code}: {r.text}"
        )

    return {"temp_token": temp_token}


@app.post("/agent/create/service_agent")
def agent_create_service_agent(config: ServiceAgentRequest) -> Dict:
    """
    A service_agent is an LLM Agent used for utilities functions such as assessing if an utterance is completed. These can use either a
    local or a 3rdparty API. Unlike YakAgents that power avatars these agents are intended to provide short, quick responses and do not have conversation memory.
    The model used is
    This endpoint creates a service agent and registers it save it to the agent_registry.
    Arguments:
        business_uid: unique id for the cafe.
        user_id: a unique id for the user supplied by authentication tool.
    Returns:
        session_id: str(uuid4): the session_id under which the agent is registered.
    """
    service_agent = None
    msg: str = ""
    ok: bool = False

    try:
        cafe: Cafe = MenuHelper.get_cafe(database, business_uid=config.business_uid)
        model_choice: ModelChoice = ModelHelper.get_model_by_id(
            database, cafe.model
        )  # TODO allow an alternative model to be specified.

        logger.info(
            f"Creating service agent for: {config.business_uid} in {config.session_id}"
        )
        service_agent = YakAgent(
            business_uid=config.business_uid,
            stream=False,
            model_choice=model_choice,
            enable_memory=False,
        )

        service_agent_registry[config.session_id] = service_agent
        logger.info(
            f"Ok. Created service agent for session_id {config.session_id}, llm:{model_choice.name}"
        )
        ok = True
    except Exception as e:
        msg = f"A problem occured while creating a yak_agent: {e}"
        logger.error(msg)
    return {"status": "success" if ok else "error", "msg": msg, "payload": ""}


@app.get("/agent/reservation/")
def agent_reservation():
    """
    Create a placeholder in the agent registry and return a session_id.
    The agent will be created when the conversatino starts.
    """
    session_id: str = str(uuid4())
    logger.info(
        f"Session ID {session_id} reserved"
    )  # TODO add time limit for session_id to expire.
    agent_registry[session_id] = None
    return {"session_id": session_id}


@app.post("/agent/create/")
def agent_create(config: SessionStart) -> Dict:
    """
    Create an instance of an Agent and save it to the agent_registry.
    Arguments:
        business_uid: unique id for the cafe.
        menu_id: unique ID of menu for the business.
        agent_rules: contains house rules, menu rules, agent personality rules as concatenated
                    list of strings. Rules are preserved in LLM context even during conversation memory pruning.
        stream: boolean indicating if response from chat should be streamed back.
        user_id: a unique id for the user supplied by authentication tool.
    Returns:
        session_id: str(uuid4): the session_id under which the agent is registered.
    """
    yak_agent = None
    msg: str = ""
    ok: bool = False

    try:
        cafe: Cafe = MenuHelper.get_cafe(database, business_uid=config.business_uid)
        menu: Menu = MenuHelper.get_one_menu(
            database, business_uid=config.business_uid, menu_id=config.menu_id
        )
        model_choice: ModelChoice = ModelHelper.get_model_by_id(database, cafe.model)

        if menu is None:
            return {
                "status": "error",
                "msg": f"Menu {config.menu_id} not found",
                "payload": "",
            }

        logger.info(f"Creating agent for: {config.business_uid} in {config.session_id}")

        if config.user_id is not None:
            raise NotImplementedError("User based customisation not yet implemented.")
        else:
            rule_set: List[str] = [_MENU_RULE_PREFIX + "\n" + menu.menu_text]
            rule_set.extend(menu.rules.split("\n"))
            rule_set.extend(cafe.house_rules.split("\n"))
            rule_set.extend(config.avatar_personality.split("\n"))
            rule_set = list(filter(lambda x: len(x.strip()) > 0, rule_set))

            # Get avatar configurations
            avatar_config: Dict = MenuHelper.parse_dict(cafe.avatar_settings)
            if "voice" in avatar_config:
                voice_id = avatar_config["voice"]
                del avatar_config["voice"]
            else:
                # Get the agent/avatar voice_id or fall back to system default.
                voice_id = app_config.text_to_speech.default_voice_id

            yak_agent = YakAgent(
                business_uid=config.business_uid,
                rules=rule_set,
                stream=config.stream,
                voice_id=voice_id,
                avatar_config=avatar_config,
                model_choice=model_choice,
                notes=cafe.notes,
            )

        agent_registry[config.session_id] = yak_agent
        logger.info(
            f"Ok. Created agent for {config.business_uid}, menu_id {config.menu_id} with session_id {config.session_id}, llm:{model_choice.name}"
        )
        ok = True
    except Exception as e:
        msg = f"A problem occured while creating a yak_agent: {e}"
        logger.error(msg)
    return {"status": "success" if ok else "error", "msg": msg, "payload": ""}


@app.get("/agent/get_avatar_config/{session_id}")
def get_avatar_config(session_id: str) -> Union[Dict, None]:
    """
    Avatar config is used for all non-voice related.
    """
    logger.info(f"Get avatar config : sesssion_id {session_id}")
    msg: str = ""
    ok: bool = True
    ret: Any = None

    # Retrieve the Agent (and agent memory) if session already underway
    session_id: str = session_id
    if session_id not in agent_registry:
        msg = f"Error: Request for agent bound to session_id: {session_id} but none exists."
        ok = False
        logger.error(msg)
        raise RuntimeError(
            "No agent found. An agent must be created prior to starting chat."
        )

    if ok:
        yak: YakAgent = agent_registry[session_id]
        try:
            ret = yak.avatar_config
        except Exception as e:
            msg = "SessionID has been reserved but agent not yet created. This can be due to the way react.js loads the app twice in dev mode to test for side effect."
            ok = False
        logger.debug(f"avatarConfig: {ret}")

    return StdResponse(ok, msg, ret)


@app.post("/chat_with_agent")
def chat_with_agent(message: ApiUserMessage) -> Union[Any, Dict[str, str]]:
    """
    Chat text_generation using griptape agent. Conversation memory is managed by Griptape so only the new question is passed in.
        Arguments:
            message: ApiUserMessage : message with session_id and opt(user_id)
        Returns:
            {'data': results}: chat completion results as dictionary. Mimics response from direct call to the models API.
        Raises:
            RuntimeException if dialogs doesn't deserialize to a valid InferernceSessionPrompt
    """

    logger.info(f"Request for text chat : sesssion_id {message.session_id}")
    logger.debug(
        f"Request for text chat : sesssion_id {message.session_id}, user_input: {message.user_input}"
    )

    # Retrieve the Agent (and agent memory) if session already underway
    session_id: str = message.session_id
    if session_id not in agent_registry:
        logger.error(
            f"Error: Request for agent bound to session_id: {message.session_id} but none exists."
        )
        raise RuntimeError(
            "No agent found. An agent must be created prior to starting chat."
        )

    yak: YakAgent = agent_registry[session_id]

    if getattr(yak, "stream"):
        logger.debug(
            f"Request for text chat : sesssion_id {message.session_id} sending to streaming response to chat_with_agent "
        )
        response = Stream(yak.agent).run(message.user_input)
        return StreamingResponse(my_gen(response), media_type="text/stream-event")
    else:
        response = yak.run(message.user_input).output.to_text()
        logger.debug(
            f"Agent for sesssion_id {message.session_id} sending to NON-streaming response to chat_with_agent "
        )
        return {"data": response}


@app.post("/get_agent_to_say")
def get_agent_to_say(message: ApiUserMessage) -> Dict:
    """
    Utility function to that gets the agent to say a short message. Not suitable for long messages as its not optimized for latency. TODO
    Because prompt submitted to say contain all the needed context, we should always try the cache if its available.
    The inner function 'stream_generator' takes a text string not a streaming response.
    @return:
        visemes and audio data.
    """
    logger.info(f"Request for /get_agent_to_say : {message.user_input}")
    # Retrieve the Agent (and agent memory) if session already underway
    session_id: str = message.session_id
    if session_id not in agent_registry:
        logger.error(
            f"Error: Request for agent bound to session_id: {session_id} but none exists."
        )
        raise RuntimeError(
            "No agent found. An agent must be created prior to starting chat."
        )

    yak: YakAgent = agent_registry[session_id]
    avatar_config: AvatarConfigParser = AvatarConfigParser(yak.avatar_config)

    # See if text is in the cache
    response, visemes, blendshapes, audio = "", [], [], ""
    if yak.usingCache and cache:
        response, visemes, blendshapes, audio = CacheUtils.get_from_cache(
            QueryType.RESPONSE, message.user_input, yak, cache
        )

    if response != "":

        def cached_stream_generator(visemes, blendshapes, audio):
            logger.debug("Yielding from cache")

            for i in range(len(visemes)):
                yield BlendShapesMultiPartResponse(
                    json.dumps(blendshapes[i]), json.dumps(visemes[i]), audio[i]
                ).prepare()

        return StreamingResponse(
            cached_stream_generator(visemes, blendshapes, audio),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    else:
        # otherwise generate it
        TTS: TextToSpeechClass = None
        if yak.TextToSpeech:
            TTS = yak.TextToSpeech
        else:
            TTS = AzureTTSViseme(
                voice_id=yak.voice_id,
                audio_config=None,
                use_blendshapes=avatar_config.blendshapes,
            )
            yak.TextToSpeech = TTS

        def stream_generator(prompt):
            with TimerContextManager(
                "SAY: GenerateAudioAndVisemes", logger, logging.DEBUG
            ):
                stream, visemes, blendshapes = TTS.audio_viseme_generator(prompt)
            yield BlendShapesMultiPartResponse(
                json.dumps(blendshapes), json.dumps(visemes), stream.audio_data
            ).prepare()
            # Cache first utterances if cache is in use
            if yak.usingCache and cache:
                CacheUtils.cache_if_short_utterance(
                    prompt,
                    cache,
                    yak,
                    stream.audio_data,
                    visemes,
                    blendshapes,
                )

        logger.debug(f"Sending streaming response, session_id {session_id}")
        return StreamingResponse(
            stream_generator(message.user_input),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )


@app.get("/get_last_response/{session_id}")
def get_last_response(session_id: str) -> Dict[str, str]:
    """
    Get the complete last response generated by the agent.
    """
    yak: YakAgent = agent_registry[session_id]
    last_response: str = ""

    try:
        last_run_index = len(yak.agent.memory.runs) - 1
        last_response = yak.agent.memory.runs[last_run_index].output
    except Exception as e:
        logger.warning(
            f"No conversation runs available for this agent. {session_id}, {e}"
        )

    logger.debug(f"Last resposne for {session_id}, {last_response} ")
    return {"last": last_response}


@app.post("/talk_with_agent")
def talk_with_agent(message: ApiUserMessage) -> Dict:
    """
    Get a synthesised voice for the stream LLM response and send that audio data back to the app.
    Does NOT generate Visemes for lipsync. See /talk_with_avatar for visemes.
    Forces streaming response regardless of Agent settings.
    """
    logger.info(f"Request spoken conversation for session_id: {message.session_id}")
    logger.debug(f"User input: {message.user_input}")

    session_id: str = message.session_id
    if session_id not in agent_registry:
        logger.error(
            f"Error: Request for agent bound to session_id: {message.session_id} but none exists."
        )
        raise RuntimeError(
            "No agent found. An agent must be created prior to starting chat."
        )

    yak: YakAgent = agent_registry[session_id]

    TTS: AzureTextToSpeech = AzureTextToSpeech(voice_id=yak.voice_id, audio_config=None)
    message_accumulator = []
    response = Stream(yak.agent).run(message.user_input)  # Streaming response.
    yak.agent_status = YakStatus.TALKING

    def stream_generator(response) -> Generator[Any, str, Any]:
        for phrase in TTS.text_preprocessor(
            response, filter=TTS.permitted_character_regex
        ):
            stream = TTS.audio_stream_generator(phrase)
            yield MultiPartResponse(json.dumps(phrase), stream.audio_data).prepare()
            if yak.status != YakStatus.TALKING:
                # status can be changed by a call from client to the /interrupt_talking endpoint.
                break
        yak.status = YakStatus.IDLE

    return StreamingResponse(
        stream_generator(response),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/agent/talk_with_avatar")
async def talk_with_avatar(message: ApiUserMessage):
    """
    Get text to speech and the viseme data for lipsync and optionally blendshapes
    Can be interrupted.
    """
    logger.info(f"Request spoken conversation for session_id: {message.session_id}")
    logger.debug(
        f"User input received: {message.user_input} @ {datetime.now().timestamp()*1000}"
    )

    logger.debug(f"TIMER: Recieved text request @ {datetime.now().timestamp()*1000}")

    session_id: str = message.session_id
    if session_id not in agent_registry:
        logger.error(
            f"Error: Request for agent bound to session_id: {message.session_id} but none exists."
        )
        raise RuntimeError(
            "No agent found. An agent must be created prior to starting chat."
        )

    yak: YakAgent = agent_registry[session_id]
    avatar_config: AvatarConfigParser = AvatarConfigParser(yak.avatar_config)

    # Check if the text and response is already cached.
    # IMPORTANT: If the text is a follow up question ( refering to revious parts of the conversation) this can lead to irrelevant responses.
    # Eventually we'll need to introduce some logic to classify if the question is refering to previous conversations turns or is standalone.
    # e.g Does the carrot cake contain nuts? vs. does it have nuts.
    # Short term solution is to filter out input text that contains pronouns such as it, they, its etc.
    cache_key: str = ""
    cached_response: Dict = None

    isCompleteUtterance: bool = True
    with TimerContextManager("COMPLETED UTTERANCE", logger, logging.DEBUG):
        service_agent: ServiceAgent = None
        if not (session_id in service_agent_registry):
            agent_create_service_agent(
                ServiceAgentRequest(
                    session_id=session_id, business_uid=yak.business_uid
                )
            )
        try:
            service_agent = service_agent_registry[session_id]
            yak.prompt_accumulator.push(message.user_input)
            completion_check: str = STTUtilities.isCompleteThought(
                input_text=yak.prompt_accumulator.prompt, service_agent=service_agent
            )
            isCompleteUtterance = completion_check["answer"]
            if isCompleteUtterance:
                message.user_input = yak.prompt_accumulator.prompt
                yak.prompt_accumulator.reset()
            else:
                logger.debug(f"Input complete:{completion_check}")
                return

        except Exception as e:
            logger.error(
                f"Error evaluating completeness for session_id: {session_id}: {e}"
            )

    if yak.usingCache and cache is not None:
        # check the cache for the key f'voice_id:::<<message.user_input>>
        cache_key = cache.get_cache_key(
            QueryType.REQUEST, yak.voice_id, message.user_input
        )
        if has_pronouns(message.user_input) == False:
            cached_response = cache.hgetall(name=cache_key)

    if cache and cached_response and len(cached_response) > 0:
        logger.info(f"Cache Hit for {cache_key}")

        def cached_stream_generator(cached_data):
            # Expects data of the form {'audio':[], 'visemes': [], 'blendshapes': []}
            # Where each element in the list is a chunk of that data previously generated by STT.
            # Data is stroed as strings in redis and so much be converted back to the proper types for blendshapes, visemes, audio
            #  and retrieved as byte strings by redis-py.
            logger.debug("Yielding from cache")
            response, blendshapes, visemes, audio = (
                cached_data[b"response"],
                pickle.loads(cached_data[b"blendshapes"]),
                pickle.loads(cached_data[b"visemes"]),
                pickle.loads(cached_data[b"audio"]),
            )
            for i in range(len(visemes)):
                if yak.status != YakStatus.TALKING:  # Have been interrupted.
                    break

                yield BlendShapesMultiPartResponse(
                    json.dumps(blendshapes[i]), json.dumps(visemes[i]), audio[i]
                ).prepare()

            yak.status = YakStatus.IDLE
            # Update the conversation memory in the yakagent so the conversation continuity can be maintained.
            yak.agent.memory.add_run(
                Run(input=message.user_input, output=response.decode("ascii"))
            )  # response is binary string due to teh way Redis stores data.

        yak.agent_status = YakStatus.TALKING

        return StreamingResponse(
            cached_stream_generator(cached_response),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    else:
        # Use the LLM to get the response back.

        TTS: TextToSpeechClass = None

        if yak.TextToSpeech:
            TTS = yak.TextToSpeech
        else:
            TTS = AzureTTSViseme(
                voice_id=yak.voice_id,
                audio_config=None,
                use_blendshapes=avatar_config.blendshapes,
            )
            yak.TextToSpeech = TTS

        response = Stream(yak.agent).run(message.user_input)

        def stream_generator(response) -> Generator[str, str, bytes]:
            full_text: List = []
            yielded_from_cache: bool = False

            for phrase, overlap in TTS.text_preprocessor(
                response, filter=TTS.permitted_character_regex, use_ssml=True
            ):
                if yak.status != YakStatus.TALKING:
                    # When being interupted, the agent_status is be forced to YakStatus.IDLE by an external function
                    logger.info("Interrupted")
                    break

                logger.debug(
                    f"TIMER: text chunk recieved @ {datetime.now().timestamp()*1000}: {phrase} + {overlap}"
                )

                response: str = ""
                blendshapes: List = None
                visemes: List = None
                audio_data: bytes = (
                    None  # Possibly compressed, depending on speech synth config
                )

                # Check in the cache for first utterance.
                # Note: cached QueryType.Responses are only use for latency reduction and so only short utterances are cached, utterances
                #       that are at most one text chunk long.
                if yak.usingCache and cache:
                    with TimerContextManager("CheckCache", logger, logger_level):
                        response, visemes, blendshapes, audio_data = (
                            CacheUtils.get_from_cache(
                                QueryType.RESPONSE, phrase, yak, cache
                            )
                        )
                    # visemes/blendshape/audio_data are all stored as lists of chunks.
                    # However, only a single chunk is acceptabled for short utterances so all others are drpped.
                    if audio_data and isinstance(audio_data, list):
                        audio_data = audio_data[0]
                        visemes = visemes[0]
                        blendshapes = blendshapes[0]

                if response == "":
                    logger.debug(f"TextChunkSentForSynth:{phrase}")
                    with TimerContextManager(
                        "GenerateAudioAndVisemes", logger, logger_level
                    ):
                        stream, visemes, blendshapes = TTS.audio_viseme_generator(
                            phrase, overlap
                        )
                    audio_data = (
                        stream.audio_data
                    )  # a bytes like object of audio. The audio might be compressed (e.g. mp3 if that's how the TTS is configured)
                    if overlap != "":
                        # overlap only has a non-empty value if we've added extra words to fix intonation problems. This should be dropped.
                        # boundary of words to be dropped from audio. Note that phrase is never modified with addtional words.
                        try:
                            truncated_duration_ms = TTS.wordboundary_log[
                                int(os.environ["WORD_COUNT_FOR_FIRST_SYNTHESIS"])
                            ]
                            with TimerContextManager(
                                "AudioTruncationTimer", logger, logger_level
                            ):
                                audio_data = TTSUtilities.truncate_audio_byte_data(
                                    audio_data, truncated_duration_ms
                                )
                        except Exception as e:
                            logger.error(f"Error truncating audio: {e} ")
                    yielded_from_cache = False
                else:
                    yielded_from_cache = True

                logger.debug(
                    f"YEILD:(phrase, source, #B,#V, time):: {'CACHE' if yielded_from_cache else 'GENERATED'}  *{phrase}* {len(blendshapes), len(visemes)} @ {datetime.now().timestamp()*1000}"
                )

                yield BlendShapesMultiPartResponse(
                    json.dumps(blendshapes), json.dumps(visemes), audio_data
                ).prepare()

                # Caching of audio/visemes/blendshapes
                if yak.usingCache and cache is not None:
                    full_text.append(phrase)
                    # Store list of chunks against full prompt
                    cache.append_to_cache(cache_key, "blendshapes", blendshapes)
                    cache.append_to_cache(cache_key, "visemes", visemes)
                    cache.append_to_cache(cache_key, "audio", audio_data)
                    logger.debug(f"Add chunks to cache")

                    if not (yielded_from_cache):
                        # If this was generated by the TTS and its short, then cache it for latency reduction.
                        # This duplicates the value in the cache, but short phrases are rare in natural language.
                        CacheUtils.cache_if_short_utterance(
                            phrase, cache, yak, audio_data, visemes, blendshapes
                        )

            if yak.usingCache:
                cache.hset(
                    cache_key, "response", "".join(full_text)
                )  # Add the full text of the response for usin in conversation memory.

            # After yeilding all the cached audio, set status to idle.
            yak.status = YakStatus.IDLE

        yak.agent_status = YakStatus.TALKING

        return StreamingResponse(
            stream_generator(response),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )


@app.get("/agent/interrupt/{session_id}")
def agent_interrupt(session_id: str):
    """Change the agent_status. If set to IDLE, this will interrupt the speech generation in the talk_with_{agent|avatar} endpoints."""
    logger.info(f"Speech interupted: session_id {session_id}")
    yak: YakAgent = agent_registry[session_id]
    if yak:
        yak.status = YakStatus.IDLE
    return StdResponse(True, "OK", "Interrupted")


"""
Misc
"""


@app.get("/services/get_ai_prompts/{businessUID}")
async def services_get_ai_prompts(businessUID: str) -> Dict:
    """Get ai prompts for text editing"""

    prompts: List[str] = ServicesHelper.get_field_by_business_id(
        database, business_uid=businessUID, field="prompts"
    )
    if prompts is not None:
        return {"status": "success", "msg": "", "payload": prompts}
    else:
        prompts = ServicesHelper.get_field_by_business_id(
            database, business_uid=_DEFAULT_BUSINESS_UID, field="prompts"
        )
        if prompts is not None:
            return {"status": "success", "msg": "", "payload": prompts}

    return {"status": "error", "msg": "No prompts found", "payload": "--none--"}


@app.post("/services/service_agent/")
def service_agent(request: ThirdPartyServiceAgentRequest) -> Dict:
    """Generic LLM model response from service_agent."""
    service_agent: ServiceAgent = None
    response = None
    ok = False
    try:
        service_agent: ServiceAgent = ServiceAgent(
            task=request.task, stream=request.stream
        )  # use defaults set on server.
        response = service_agent.do_job(request.prompt)
        ok = True
    except Exception as e:
        response = f"Error invoking service_agent: {e}"
        logger.error(response)

    if ok:
        if request.stream:
            return StreamingResponse(response)
        else:
            return {"status": "success", "msg": response}
    else:
        return {"status": "error", "msg": response}


"""
Deal with menus
"""


@app.post("/menus/upload/")
async def upload_menu(
    business_uid: str = Form(...),
    file: UploadFile = File(...),
    grp_id: Optional[str] = Form(...),
):
    """Save menu image to disk and add path to database. Returns the uuid of the menu and a collection_id for grouping multiple pages"""

    # TODO validate file.
    # Check if the file is a PNG image
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(
            status_code=400, detail="File must be an image/png or image/jpeg"
        )

    file_extension = Path(file.filename).suffix
    if file_extension not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file extension. Only png,jpeg, jpg accepted.",
        )

    file_id = str(uuid.uuid4())
    file_path = f"{app_config.assets.image_folder}/{file_id}{file_extension}"

    # create thumbnail to avoid sending large files back to client
    content = await file.read()
    image_stream = io.BytesIO(content)
    raw_image = Image.open(image_stream)
    raw_image.save(file_path)
    image_stream.seek(0)
    AR = raw_image.width / raw_image.height
    lower_res_size = (
        math.floor(app_config.assets.thumbnail_image_width * AR),
        app_config.assets.thumbnail_image_width,
    )
    lowres_image = raw_image.resize(lower_res_size, Image.LANCZOS)
    lowres_file_path = (
        f"{app_config.assets.image_folder}/{file_id}_lowres{file_extension}"
    )
    lowres_image.save(lowres_file_path)

    # Create a Menu object with menu_id set to the file_id
    sequence_number: int = 0
    # Check for null-like values tha may occur when frontend passes non-initialized grp_id
    if grp_id is None or grp_id == "null" or grp_id == "":
        # This is a new collection so we need to create a collection id.
        _grp_id = str(uuid.uuid4())
    else:
        _grp_id = grp_id
        sequence_number = MenuHelper.count_menus_in_collection(
            database, business_uid, grp_id
        )  # use as base-0 sequnce for images in the same collection

    new_menu: Menu = Menu(
        menu_id=file_id,
        collection={"grp_id": _grp_id, "sequence_number": sequence_number},
        raw_image_rel_path=f"{file_id}{file_extension}",
        thumbnail_image_rel_path=f"{file_id}_lowres{file_extension}",
    )

    # Create or update the cafe with the new menu
    ok, msg = MenuHelper.save_menu(database, business_uid, new_menu)

    return {
        "status": "success" if ok else "erorr",
        "message": msg,
        "payload": {"menu_id": new_menu.menu_id, "grp_id": _grp_id},
    }


@app.get("/menus/collate_images/{business_uid}/{grp_id}")
def menus_collate_images(business_uid: str, grp_id: str):
    """Collate the text from all the menu partial images belonging to the collection identified by grp_id"""
    ok = False
    msg: str = ""
    count: int = -1
    primary_menu_id: str = ""

    ok, msg, count, primary_menu_id = MenuHelper.collate_text(
        database, business_uid, grp_id
    )
    if ok:
        logger.info(f"Collated text from {count} images into menu_id {primary_menu_id}")
    else:
        logger.error(
            f"Failure during collation of text from multiple images for menu.collection.grp_id {grp_id}"
        )

    return {
        "status": "success" if ok else "erorr",
        "message": msg,
        "payload": {"primary_menu_id": primary_menu_id},
    }


@app.get("/menus/get_one/{business_uid}/{menu_id}")
async def menus_get_one(business_uid: str, menu_id: str):
    menu: Menu = MenuHelper.get_one_menu(database, business_uid, menu_id)
    # if menu is not None:
    #    menu = Helper.insert_images(config, menu)
    return {
        "status": "success",
        "msg": "",
        "menu": menu.to_dict() if menu is not None else None,
    }


@app.get("/menus/get_all/{business_uid}")
async def menus_get_all(business_uid: str, for_display: bool = True):
    """Get all the menus"""
    menus: List[Menu] = MenuHelper.get_menu_list(database, business_uid)
    if len(menus) == 0:
        # It might just be that there are none.
        return {
            "status": "warning",
            "message": f"Failed getting menu list for business {business_uid}",
            "menus": [],
        }
    else:
        # Insert thumbnail image data into the menu records before sending to client.
        loaded_menus = MenuHelper.insert_images(
            app_config, menus=menus, image_types=[ImageSelector.THUMBNAIL]
        )
        if loaded_menus is None:
            return {"status": "Warning", "message": "No thumbnail menus returned."}

    return {
        "status": "success",
        "message": "",
        "menus": [menu.to_dict() for menu in loaded_menus],
    }


@app.get("/menus/get_as_options/{business_uid}/{encoded_utc_time}")
def menus_get_as_options(business_uid: str, encoded_utc_time: str):
    """Get all menus and choose a default based on the menu time of day validity and the passed in time.
    Note:
        All dates in Yak are stored as UTC time. Conversion to local time is consumer responsibility
        encoded_utc_time may not contain the postfix Z but it wall alwasy be assumed to be in UTC time.
    """

    menus: List[Menu] = MenuHelper.get_menu_list(
        database, business_uid, for_display=True
    )
    decoded_utc_time: str = urllib.parse.unquote(encoded_utc_time).rstrip(
        "Z"
    )  # Drop the ISO 8601 explict maker for UTC time.
    utc_time: datetime = datetime.fromisoformat(decoded_utc_time)  # UTC time
    default_menu_id: str = ""
    options: List[Dict[str, str]] = []
    msg: str = ""

    if len(menus) > 0:  # Get the first valid one and make it the default.
        for menu in menus:
            options.append({"label": menu.name, "value": menu.menu_id})
            if "start" in menu.valid_time_range and "end" in menu.valid_time_range:
                # Check if time of day falls within the valid time range.
                if (
                    menu.valid_time_range["start"].date()
                    != menu.valid_time_range["end"]
                ):
                    # time range straddles 2 different days.
                    if (
                        utc_time.time() <= menu.valid_time_range["end"].time()
                        or utc_time.time() >= menu.valid_time_range["start"].time()
                    ):
                        default_menu_id = menu.menu_id
                        break
                else:  # same day
                    if (
                        utc_time.time() <= menu.valid_time_range["end"].time()
                        and utc_time.time() >= menu.valid_time_range["start"].time()
                    ):
                        default_menu_id = menu.menu_id
                        break
    else:
        msg = "No menus returned."
        logger.warning(f"No menus found for business_uid = {business_uid}")

    return {
        "status": "success" if len(menus) > 0 else "warning",
        "msg": msg,
        "payload": {"options": options, "default": default_menu_id},
    }


@app.put("/menus/update_one/{business_uid}/{menu_id}")
async def menus_update_one(business_uid: str, menu_id: str, menu: Menu):
    """Update one menu in the cafe.menus. Menu contains optional fields, which, when absent leave the stored menu field unchanged."""
    ok, msg = MenuHelper.update_menu(database, business_uid, menu)
    return {"status": "success" if ok == True else "error", "message": msg}


@app.get("/menus/delete_one/{business_uid}/{menu_id}")
async def menus_delete_one(business_uid: str, menu_id: str):
    ok, msg = MenuHelper.delete_one_menu(database, business_uid, menu_id)
    return {"status": "success" if ok == True else "error", "message": msg}


@app.get("/menus/ocr/{business_uid}/{menu_id}")
async def menu_ocr(business_uid: str, menu_id: str):
    """Call the tesseract OCR endpoint see https://github.com/hertzg/tesseract-server"""

    ret = None
    status: bool = False
    url: str = app_config.ocr.url
    data = {
        "options": json.dumps(
            {
                "languages": ["eng"],
                "dpi": 300,
            }
        )
    }
    # Need the file extension
    menu: Menu = MenuHelper.get_one_menu(
        database, business_uid=business_uid, menu_id=menu_id
    )
    file_path = f"{app_config.assets.image_folder}/{menu_id}.png"

    try:
        # Tesseract requires the file object to be passed in not the URL.
        with open(file_path, "rb") as fp:
            files = {"file": fp}
            response = requests.post(url, data=data, files=files)
            if response.ok:
                ret = json.loads(response.text)
                if (
                    "stdout" in ret["data"]
                ):  # contains messages. OCR text in response.content.stdout
                    # Save it to db
                    status, msg = MenuHelper.update_menu_field(
                        database,
                        business_uid,
                        menu_id,
                        ret["data"]["stdout"],
                        "menu_text",
                    )
                else:
                    logger.error(
                        f'menu_ocr has not field "stdout" business {business_uid}: err {msg}'
                    )
    except Exception as e:
        msg = e
        logger.error(f"Error in performing OCR. Message {e}")

    return {"status": "success" if status == True else "error", "message": msg}


"""
Business entities
"""


@app.get("/cafe/settings/get/{business_uid}")
def get_settings(business_uid: str):
    cafe = MenuHelper.get_cafe(database, business_uid=business_uid)
    if cafe is not None:
        cafe.menus = []  # Not needed but must NOT be None
        return StdResponse(True, "", cafe.to_dict()).to_dict()
    else:
        return StdResponse(False, "No cafe returned. Please check logs.", {}).to_dict()


@app.post("/cafe/settings/save")
def cafe_save_settings(settings: Cafe):
    ok: bool = False
    msg: str = ""
    ret = None

    ok, msg = MenuHelper.upsert_cafe_settings(database, settings.business_uid, settings)

    return StdResponse(ok, msg).to_dict()


@app.get("/data/options/{business_uid}/{table_name}/{columns}")
def cafe_get_setting_options(business_uid: str, table_name: str, columns: str):
    ok: bool = False
    msg: str = ""
    ret = None
    return_fields: str = columns.split(",")

    data: List[Dict] = DataHelper.get_non_business_data(
        database, table_name=f"{table_name}", return_field_names=return_fields
    )
    if data is not None:
        ok = True
        data = list(data)
    else:
        msg = f"Failed to retrieve fields {return_fields} for {table_name}"

    return StdResponse(ok, msg, data).to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    load_dotenv()  # load all the environment variables

    app_config = OmegaConf.load(
        os.path.join(
            os.environ["APPLICATION_ROOT_FOLDER"],
            os.environ["API_CONFIG_PATH"],
            "configs.yaml",
        )
    )

    # Instantiate a Mongo class that provides API for pymongo interaction with mongodb.
    database = DatabaseConfig(app_config)

    # Redis is used for caching LLM, and TTS results for improved latency.
    if os.environ["USE_API_CACHE"]:
        cache = RedisHelper()
    else:
        cache = None

    logger = logging.getLogger("YakChatAPI")
    logger_level = logging.DEBUG
    logger.setLevel(logger_level)

    log_file_path = os.path.join(app_config.logging.root_folder, "session_logs.log")
    createIfMissing(log_file_path)

    file_handler = RotatingFileHandler(
        log_file_path, mode="a", maxBytes=1024 * 1024, backupCount=15
    )
    file_handler.setLevel(logger_level)

    # Create formatters and add it to handlers
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    logger.addHandler(file_handler)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=app_config.api.port)
