from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Callable
from dotenv import load_dotenv
from attrs import define, field

from griptape.structures import Agent
from griptape.utils import Chat, PromptStack
from griptape.drivers import HuggingFaceInferenceClientPromptDriver
from griptape.events import CompletionChunkEvent, FinishStructureRunEvent
from griptape.rules import Rule, Ruleset

from transformers import AutoTokenizer
import os
import json
import logging

logger = logging.getLogger(__name__)

@define(kw_only=True)
class YakAgent:
    """
        Helper class for creating Chat Agent.

        Attributes:
            model: HF namespace/repo_id or URL of model_endpoint.
            tokenizer: Hf namespace/repo_id or path to locally saved AutoTokenizer
            stream: boolean indicating if the chat response should be streamed back. 
            stream_chunk_size: the number of chunks to accumulate when streaming before yielding back to client.
            token: HF token. Not needed if serving model locally.
            params: Dictionary of model/tokenizer specific parameters.
    """
    model:str = field(default='http://localhost:8080')
    tokenizer:str = field(default='/home/mtman/Documents/Repos/yakwith.ai/models/Mistral-7B-OpenOrca')
    max_tokens: Optional[int] = field(default=1024)
    temperature: Optional[float] = field(default=0.9)
    stream : Optional[bool] = field(default=False)
    stream_chunk_size: Optional[int] = field(default=1)
    token: Optional[str] = field(default='DUMMY')
    rule_set: Optional[Ruleset] = field(default=None)

    # Task specific parameters see SampleParameters(), TODO check the duplications here. 
    params: Optional[Dict]= field(
        default={
                "stop_sequences":['</s>','<s>','<|im_end|>','<|im_start|>'],
                "max_new_tokens":1024
                }
            )
    
    callbacks: Optional[List[Callable]] = field(init=False)
    output_fn: Optional[Callable] = field(init=False)

    def __attrs_post_init__(self):
        if self.stream:
            self.callbacks = {
                    CompletionChunkEvent:[lambda x: print(x.token, end='')],
                    FinishStructureRunEvent: [lambda _: print('\n')]
                    }
            self.output_fn = lambda x: x
        else:
            self.callbacks = None
            self.output_fn = lambda x: print(x)

        self.agent = Agent(prompt_driver = HuggingFaceInferenceClientPromptDriver(
                        model = 'http://localhost:8080/',
                        pretrained_tokenizer = self.tokenizer,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        params=self.params,
                        token=self.token,
                        stream=self.stream,
                        stream_chunk_size=self.stream_chunk_size,
                        ),
                        event_listeners=self.callbacks,
                        logger_level=logging.ERROR,
                        ruleset = self.rule_set
                        #tools = [WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])],
                )
    
    def run(self,*args,**kwargs):
        return self.agent.run(*args,**kwargs)