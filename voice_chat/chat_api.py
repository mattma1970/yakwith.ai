from fastapi import FastAPI, Response, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import requests
import uuid
import shutil
from pathlib import Path
import base64
import math
import urllib

from PIL import Image
import io

from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any, Iterator, Tuple
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

from voice_chat.yak_agents import YakAgent, YakStatus, ServiceAgent
from voice_chat.data_classes.chat_data_classes import (
    ApiUserMessage,
    AppParameters,
    SessionStart,
    SttTokenRequest,
    ServiceAgentRequest,
)
from voice_chat.data_classes.data_models import Menu, Cafe, ImageSelector
from voice_chat.data_classes.mongodb_helper import (
    MenuHelper,
    DatabaseConfig,
    ServicesHelper,
)

from bson import ObjectId

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
_DEFAULT_BUSINESS_UID = "all"

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

agent_registry = {}  # Used to store one agent per session.


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
def service_agent(request: ServiceAgentRequest) -> Dict:
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
    file_path = f"{config.assets.image_folder}/{file_id}{file_extension}"

    # create thumbnail to avoid sending large files back to client
    content = await file.read()
    image_stream = io.BytesIO(content)
    raw_image = Image.open(image_stream)
    raw_image.save(file_path)
    image_stream.seek(0)
    AR = raw_image.width / raw_image.height
    lower_res_size = (
        math.floor(config.assets.thumbnail_image_width * AR),
        config.assets.thumbnail_image_width,
    )
    lowres_image = raw_image.resize(lower_res_size, Image.LANCZOS)
    lowres_file_path = f"{config.assets.image_folder}/{file_id}_lowres{file_extension}"
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
async def menus_get_all(business_uid: str, for_display:bool = True):
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
            config, menus=menus, image_types=[ImageSelector.THUMBNAIL]
        )
        if loaded_menus is None:
            return {"status": "Warning", "message": "No thumbnail menus returned."}

    return {
        "status": "success",
        "message": "",
        "menus": [menu.to_dict() for menu in loaded_menus],
    }

@app.get("/menus/get_as_options/{business_uid}/{encoded_utc_time}")
def menus_get_as_options( business_uid: str, encoded_utc_time: str):
    """ Get all menus and choose a default based on the menu time of day validity and the passed in time.
        Note: 
            All dates in Yak are stored as UTC time. Conversion to local time is consumer responsibility
            encoded_utc_time may not contain the postfix Z but it wall alwasy be assumed to be in UTC time.
    """

    menus: List[Menu] = MenuHelper.get_menu_list(database, business_uid, for_display=True)
    decoded_utc_time: str = urllib.parse.unquote(encoded_utc_time).rstrip('Z') # Drop the ISO 8601 explict maker for UTC time.
    utc_time: datetime = datetime.fromisoformat(decoded_utc_time) #UTC time
    default_menu_id: str = ''
    options: List[Dict[str,str]] = []
    msg: str = ''

    if len(menus)>0: # Get the first valid one and make it the default.
        for menu in menus:
            options.append({'label':menu.name, 'value':menu.menu_id})
            if 'start' in menu.valid_time_range and 'end' in menu.valid_time_range:
                # Check if time of day falls within the valid time range. 
                if menu.valid_time_range['start'].date()!=menu.valid_time_range['end']:
                    # time range straddles 2 different days.
                    if utc_time.time()<=menu.valid_time_range['end'].time() or utc_time.time()>=menu.valid_time_range['start'].time():
                        default_menu_id = menu.menu_id
                        break
                else: # same day 
                    if utc_time.time()<=menu.valid_time_range['end'].time() and utc_time.time()>=menu.valid_time_range['start'].time():
                        default_menu_id = menu.menu_id
                        break
    else:
        msg = 'No menus returned.'
        logger.warning(f'No menus found for business_uid = {business_uid}')

    return {
        'status': 'success' if len(menus)>0 else 'warning',
        'msg': msg,
        'payload': {'options':options,'default':default_menu_id} 
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
    url: str = config.ocr.url
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
    file_path = f"{config.assets.image_folder}/{menu_id}.png"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/mtman/Documents/Repos/yakwith.ai/voice_chat/configs/api/configs.yaml",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # Instantiate Mongo class that provides API for pymongo interaction with mongodb.
    database = DatabaseConfig(config)

    logger = logging.getLogger("YakChatAPI")
    logger.setLevel(logging.DEBUG)

    log_file_path = os.path.join(config.logging.root_folder, "session_logs.log")
    file_handler = RotatingFileHandler(
        log_file_path, mode="a", maxBytes=1024 * 1024, backupCount=15
    )
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    logger.addHandler(file_handler)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.api.port)
