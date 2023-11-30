from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import requests

from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any, Iterator, Tuple
from dotenv import load_dotenv
from uuid import uuid4
from attr import define, field
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

from voice_chat.yak_agent.yak_agent import YakAgent, YakStatus
from voice_chat.data_classes.chat_data_classes import (
    ApiUserMessage,
    AppParameters,
    SessionStart,
    SttTokenRequest,
)

from voice_chat.utils import DataProxy
from voice_chat.service.azure_TTS import AzureTextToSpeech
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

app = FastAPI()
"""
    Deal with CORS issues of browser calling browser from different ports or names.
    https://fastapi.tiangolo.com/tutorial/cors/
"""
origins = ["http://localhost", "http://localhost:3000", "https://app.yakwith.ai"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_registry = {}


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
    # TODO Check that the request has a valid client authorization token
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
    r = requests.post(
        url=service_config["api_endpoint"],
        headers=headers,
        data=json.dumps({"expires_in": service_config["duration"]}),
    )

    if r.status_code == 200:
        logger.info("OK: Got temp STT token.")
        response = r.json()
        temp_token = response["token"]

    return {"temp_token": temp_token}


@app.post("/create_agent_session")
def create_agent_session(config: SessionStart) -> Dict:
    """
    Create an instance of an Agent and save it to the agent_registry.
    Arguments:
        cafe_id: unique id for the cafe. Used to index menu, cafe policies
        agent_rules: list of rules names for agent and restaurant
        stream: boolean indicating if response from chat should be streamed back.
        user_id: a unique id for the user supplied by authentication tool.
    Returns:
        session_id: str(uuid4): the session_id under which the agent is registered.
    """
    yak_agent = None
    session_id = None

    logger.info(f"Create agent session for: {config.cafe_id}")
    if config.user_id is not None:
        raise NotImplementedError("User based customisation not yet implemented.")
    else:
        session_id = str(uuid4())
        yak_agent = YakAgent(
            cafe_id=config.cafe_id, rule_names=config.rule_names, stream=config.stream
        )

    agent_registry[session_id] = yak_agent
    logger.info(f"Ok. Created agent for {config.cafe_id} with session_id {session_id}")
    return {"session_id": session_id}


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
            f"Request for text chat : sesssion_id {message.session_id} sending to streaming response to chat_with_agent request."
        )
        response = Stream(yak.agent).run(message.user_input)
        return StreamingResponse(my_gen(response), media_type="text/stream-event")
    else:
        response = yak.run(message.user_input).output.to_text()
        logger.debug(
            f"Agent for sesssion_id {message.session_id} sending to NON-streaming response to chat_with_agent request."
        )
        return {"data": response}


@app.post("/get_agent_to_say")
def get_agent_to_say(message: ApiUserMessage) -> Dict:
    """
    Utility function to that gets the agent to say a particular message
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

    TTS: AzureTextToSpeech = AzureTextToSpeech(audio_config=None)

    def stream_generator(prompt):
        stream = TTS.audio_stream_generator(prompt)
        yield stream.audio_data  # Byte data

    logger.debug(f"Sending streaming response, session_id {session_id}")
    return StreamingResponse(
        stream_generator(message.user_input), media_type="audio/mpeg"
    )


@app.post("/talk_with_agent")
def talk_with_agent(message: ApiUserMessage) -> Dict:
    """
    Get a synthesised voice for the stream LLM response and send that audio data back to the app.
    Forces streaming response regardless of Agent sessions.
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

    TTS: AzureTextToSpeech = AzureTextToSpeech(audio_config=None)
    message_accumulator = []
    response = Stream(yak.agent).run(message.user_input)

    def stream_generator(response) -> Tuple[Any, str]:
        for phrase in TTS.text_preprocessor(response, filter="[^a-zA-Z0-9,. ]"):
            stream = TTS.audio_stream_generator(phrase)
            yield stream.audio_data  # Byte data for whole sentance and the phrase

    return StreamingResponse(stream_generator(response), media_type="audio/mpeg")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Be far more chatty about the internals.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="/home/mtman/Documents/Repos/yakwith.ai/voice_chat/logs",
        help="Root folder for the api logs",
    )

    args = parser.parse_args()

    logger = logging.getLogger("YakChatAPI")
    logger.setLevel(logging.DEBUG)

    log_file_path = os.path.join(args.log_root, "session_logs.log")
    file_handler = RotatingFileHandler(
        log_file_path, mode="a", maxBytes=1024 * 1024, backupCount=15
    )
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    logger.addHandler(file_handler)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8884)
