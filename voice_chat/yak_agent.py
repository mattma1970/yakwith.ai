from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Callable
from dotenv import load_dotenv
from attrs import define, field
from queue import Queue

from griptape.structures import Agent
from griptape.utils import Chat, PromptStack
from griptape.drivers import HuggingFaceInferenceClientPromptDriver, OpenAiChatPromptDriver
from griptape.events import CompletionChunkEvent, FinishStructureRunEvent, EventListener
from griptape.rules import Rule, Ruleset

from transformers import AutoTokenizer
import os
import json
import logging
from omegaconf import OmegaConf

from data.chat_data_classes import ModelDriverConfiguration, RuleList

logger = logging.getLogger(__name__)

load_dotenv()

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
    model_driver_config: Optional[ModelDriverConfiguration] = field(default=None)
    agent_rules : Optional[RuleList] = field(default=None)
    model_id: Optional[str] = field(default='voice_chat/configs/model_driver/default_model_driver.yaml')
    agent_rules_id: Optional[str] = field(default='voice_chat/configs/rules/agent/default_rule_set.yaml')
    user_id: Optional[str] = field(default=None)
  
    stream: Optional[bool] = field(default=False)
    streaming_event_listeners: Optional[List[EventListener]] = field(init=False)
    output_fn: Optional[Callable] = field(init=False)
    agent: Agent = field(init=False)

    def __attrs_post_init__(self):
        try:
            if self.model_driver_config is None:
                self.model_driver_config = ModelDriverConfiguration.from_omega_conf(OmegaConf.load(self.model_id))
            if self.agent_rules is None:
                self.agent_rules = RuleList.from_omega_conf(OmegaConf.load(self.agent_rules_id))
        except Exception as e:
            raise RuntimeError(f'Error loading agent related config file: {e}')
        if self.stream:
            self.streaming_event_listeners = [
                    EventListener(lambda x: print(x.token, end=''), event_types=[CompletionChunkEvent]),
                    EventListener(lambda _: print('\n'),event_types=[FinishStructureRunEvent])
            ]
            self.output_fn = lambda x: x
        else:
            self.streaming_event_listeners = []
            self.output_fn = lambda x: print(x)

        self.agent = Agent(prompt_driver = HuggingFaceInferenceClientPromptDriver(
                                                token=self.model_driver_config.token,
                                                model = self.model_driver_config.model,
                                                pretrained_tokenizer = self.model_driver_config.pretrained_tokenizer,
                                                params=self.model_driver_config.params,
                                                task = self.model_driver_config.task,
                                                stream=self.model_driver_config.stream,
                                                stream_chunk_size=self.model_driver_config.stream_chunk_size,
                                                ),
                        event_listeners=self.streaming_event_listeners,
                        logger_level=logging.ERROR,
                        rules= self.agent_rules.rules,
                        #tools = [WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])],
        )
        #self.agent = Agent(prompt_driver=OpenAiChatPromptDriver(model='gpt-3.5-turbo', stream=True),stream=True,event_listeners=self.callbacks)

    def run(self,*args,**kwargs):
        return self.agent.run(*args,**kwargs)
    
if __name__=='__main__':
    yak = YakAgent()

    from griptape.utils import Chat
    Chat(yak).start()