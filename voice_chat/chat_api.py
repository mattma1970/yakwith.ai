from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import requests

from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any, Iterator
from dotenv import load_dotenv
from uuid import uuid4
from attr import define, field
import argparse
import os
import json
import logging
from datetime import time, datetime

from voice_chat.yak_agent.yak_agent import YakAgent
from voice_chat.data_classes.chat_data_classes import ApiUserMessage, AppParameters, SessionStart, SttTokenRequest
from voice_chat.utils import DataProxy

from griptape.structures import Agent
from griptape.utils import Chat, PromptStack
from griptape.drivers import HuggingFaceInferenceClientPromptDriver
from griptape.events import CompletionChunkEvent, FinishStructureRunEvent
from griptape.rules import Rule, Ruleset
from griptape.utils import Stream
from griptape.artifacts import TextArtifact
from griptape.memory.structure import Run

from omegaconf import OmegaConf, DictConfig

from dataclasses import dataclass

_ALL_TASKS=['chat_with_agent:post','chat:post','llm_params:get']

app=FastAPI()
'''
    Deal with CORS issues of browser calling browser from different ports or names.
    https://fastapi.tiangolo.com/tutorial/cors/
'''
origins = [
            'http://localhost',
            'http://localhost:3000'
           ]
app.add_middleware(
                    CORSMiddleware,
                    allow_origins = origins,
                    allow_credentials = True,
                    allow_methods = ['*'],
                    allow_headers = ["*"] )

agent_registry = {}
logger = logging.getLogger(__name__)


def my_gen(response: Iterator[TextArtifact])->str:
    for chunk in response:
        yield chunk.value


@app.post('/get_temp_token')
def get_temp_token(req: SttTokenRequest) -> Dict:
    '''
    Get time temporary token api token for requested stt service based on the service_configurations data. 
    '''
    temp_token = None
    print('made it to the api')
    # TODO Check that the request has a valid client authorization token
    try:
        service_config = DataProxy.get_3p_service_configs(
            authorization_key=req.client_authorization_key,
            authorization_data_source_name='authorized_clients',
            service_name=req.service_name,
            service_data_source_name='service_configs'  #assemblyai_temporary_token
        )
    except Exception as e:
        raise RuntimeError(f'Failed to get service token. {req.service_name} {e}')
    
    headers = {
                'Content-Type': 'application/json',
                'Authorization': service_config['api_token']
            }
    r = requests.post(
        url=service_config['api_endpoint'],
        headers=headers,
        data=json.dumps({"expires_in":service_config['duration']})
    )
    
    if r.status_code == 200:
        response = r.json()
        temp_token = response['token']
    
    return {'temp_token':temp_token}


@app.post('/create_agent_session/')
def create_agent_session(config: SessionStart)->Dict:
    '''
    Create an instance of an Agent and save it to the agent_registry.
    Arguments:
        cafe_id: unique id for the cafe. Used to index menu, cafe policies
        agent_rules: list of rules names for agent and restaurant
        stream: boolean indicating if response from chat should be streamed back.
        user_id: a unique id for the user supplied by authentication tool.
    Returns:
        session_id: str(uuid4): the session_id under which the agent is registered.
    '''
    yak_agent = None
    session_id = None

    if config.user_id is not None:
        raise NotImplementedError('User based customisation not yet implemented.')
    else:
        session_id = str(uuid4())
        yak_agent = YakAgent(cafe_id=config.cafe_id,
                             rule_names=config.rule_names,
                             stream=config.stream)

    agent_registry[session_id]=yak_agent
    return {'session_id':session_id}

@app.post('/chat_with_agent')
def chat_using_agent(message: ApiUserMessage) -> Union[Any,Dict[str,str]]:
    '''
        Chat text_generation using griptape agent. Conversation memory is managed by Griptape so only the new question is passed in. 

        Arguments:
            message: ApiUserMessage : message with session_id and opt(user_id)
        Returns:
            {'data': results}: chat completion results as dictionary. Mimics response from direct call to the models API.
        Raises:
            RuntimeException if dialogs doesn't deserialize to a valid InferernceSessionPrompt
    '''

    # Retrieve the Agent (and agent memory) if session already underway
    session_id:str = message.session_id
    if session_id not in agent_registry:
        raise RuntimeError('No agent found. An agent must be created prior to starting chat.')
    
    yak: YakAgent = agent_registry[session_id]

    if getattr(yak, 'stream'):
        response = Stream(yak.agent).run(message.user_input)
        return StreamingResponse(my_gen(response), media_type='text/stream-event')
    else:
        response = yak.run(message.user_input).output.to_text()
        return {'data':response}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action='store_true', default =False, help='Be far more chatty about the internals.')
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8884)