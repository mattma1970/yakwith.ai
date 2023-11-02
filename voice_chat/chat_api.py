from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from dotenv import load_dotenv
from uuid import uuid4
from attr import define, field
import argparse
import os
import json
import logging

from yak_agent import YakAgent
from chat_data_classes import ApiUserMessage, AppParameters

from griptape.structures import Agent
from griptape.utils import Chat, PromptStack
from griptape.drivers import HuggingFaceInferenceClientPromptDriver
from griptape.events import CompletionChunkEvent, FinishStructureRunEvent
from griptape.rules import Rule, Ruleset
from griptape.utils import Stream

from copy import deepcopy


_ALL_TASKS=['chat_with_agent:post','chat:post','llm_params:get']

app=FastAPI()
agent_registry = {}
logger = logging.getLogger(__name__)

_agent = None

@app.get('/create_agent_session/')
def create_agent_session(user_uid: Optional[str]=None):
    '''
    Create an instance of an Agent and save it to the agent_registry.
    Arguments:
        user_id: a unique id for the user supplied by authentication tool.
    Returns:
        session_id: str(uuid4): the session_id under which the agent is registered.
    '''
    agent = None
    session_id = None

    if user_uid is not None:
        raise NotImplementedError('User based customisation not yet implemented.')
    else:
        agent = YakAgent()    
        session_id = str(uuid4())

    agent_registry['1']=agent
    agent_registry['2']=YakAgent()

    #_agent = deepcopy(agent)
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
        return StreamingResponse(response, media_type='text/plain')
    else:
        response = yak.run(message.user_input).output.to_text()
        return {'data':response}

@app.get('/llm_params')
def get_llm_params():
    '''Get some model params that are needed by the app for chat history management.'''
    return {
            'max_seq_len':args.max_seq_len,
            'max_gen_len': args.max_gen_len,
            'top_p':args.top_p,
            'max_batch_size':args.max_batch_size,
            'supported_tasks': _ALL_TASKS
            }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',type=str, default='/home/mtman/Documents/Repos/llama/')
    parser.add_argument('--model_path', type=str, default='llama-2-7b', help='Relative path to root_path')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer.model')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.4)
    parser.add_argument('--max_seq_len',type=int, default=3000)
    parser.add_argument('--max_gen_len',type=int, default=256)
    parser.add_argument('--max_batch_size',type=int, default=4)
    parser.add_argument('--debug',action='store_true', default =False, help='Be far more chatty about the internals.')
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8884)