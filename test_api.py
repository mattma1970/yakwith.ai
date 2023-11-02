from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from typing import Any, Dict

from griptape.structures import Agent
from griptape.drivers import OpenAiChatPromptDriver
from griptape.events import CompletionChunkEvent, FinishStructureRunEvent
from griptape.utils import Stream
import uvicorn
import os

import uvicorn

load_dotenv('voice_chat/.env')

app=FastAPI()

def gen(resp):
    for i in resp:
        yield i
        
@app.post('/chat_with_agent')
def chat_using_agent(message: Dict[str,str]) -> Any:
    agent.event_listeners = {CompletionChunkEvent:[lambda x: print (x.token+'-',end="")]}
    resp =Stream(agent).run(message['user_input'])
    return StreamingResponse(resp)


if __name__=='__main__':     
    callbacks = {CompletionChunkEvent:[lambda x: print (x.token+'-',end="")]}
    agent = Agent(
                    prompt_driver=OpenAiChatPromptDriver(model='gpt-3.5-turbo',stream=True),
                    event_listeners= callbacks,
                    stream=True,
            )
    uvicorn.run(app, host='0.0.0.0', port=8888)