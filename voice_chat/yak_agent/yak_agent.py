from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Callable
from dotenv import load_dotenv
from attrs import define, field, Factory, validators
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

from voice_chat.data_classes import ModelDriverConfiguration, RuleList

logger = logging.getLogger(__name__)

load_dotenv()

MODEL_DRIVER_ROOT_PATH = 'voice_chat/configs/model_driver'
RULES_ROOT_PATH = 'voice_chat/cafe_data'
RULES_AGENT_FOLDER = 'agent'
RULE_RESTAURANT_FOLDER = 'restaurant'
RULES_PARENT_FOLDER = 'rules'

@define(kw_only=True)
class YakAgent:
    """
        Helper class for creating Yak Chat Agent powered by custom griptape prompt driver HuggingFaceInferenceClientPromptDriver
        See dev branch https://github.com/mattma1970/griptape/tree/yak_dev

        YakAgent configures the agent rules, prompt driver for LLM backend

        Attributes:
            cafe_id: A uid for the establishment. Primarily used to index configurations.
            model_driver_config: ModelDriverConfiguration object for prompt driver.
            model_driver_config_name: filename of the yaml configuration file for prompt driver. If model_driver_config is specified, this is ignored.
  
            rule_names: Rules are distributed for Yak and this Dict contains the leaf folder and yaml filenames of the collection of rules.
            rules : A RuleList used to collect the distributed rules. If rules are passed in the rule_names is ignored.
            
            user_id: a uid for the user if provided. Reserved for personalisation feature in the future.
            task: the function name of the InferenceClient task that will be served.
            stream: streaming flag for both the prompt driver and the Agent.

            :: Attributes not initialized ::
            streaming_event_listeners: List[EventListener], optional: List of event listeners for the griptape.agent. Mostly useful for debugging.
            output_fn: Callable, optional: output of streaming responses.
            agent: Agent

    """
    def check_keys(self, attribute, value):
        if value is not None:
            for key,_ in value.items():
                if key not in [RULES_AGENT_FOLDER, RULE_RESTAURANT_FOLDER]:
                    raise ValueError(f'Only {RULES_AGENT_FOLDER},{RULE_RESTAURANT_FOLDER} are permitted keys for rules config file locations.')
                
    cafe_id: str =field(default='default', kw_only=True)
    model_driver_config_name: Optional[str] = field(default='default_model_driver')
    model_driver_config: Optional[ModelDriverConfiguration] = field(default=None)
    rule_names: Optional[Dict] = field(default=Factory(dict), validator=[check_keys])
    rules : Optional[list[Rule]] = field(default=Factory(list))
    user_id: Optional[str] = field(default=None)
  
    task: Optional[str] = field(default='text_generation', kw_only=True)
    stream: Optional[bool] = field(default=False)
    streaming_event_listeners: Optional[List[EventListener]] = field(default=None, kw_only=True)
    output_fn: Optional[Callable] = field(init=False)
    agent: Agent = field(init=False)

    def __attrs_post_init__(self):
        try:
            if self.model_driver_config is None:
                if '.yaml' not in self.model_driver_config_name:
                    self.model_driver_config_name+='.yaml'
                config_filename=os.path.join(MODEL_DRIVER_ROOT_PATH,self.model_driver_config_name)
                self.model_driver_config = ModelDriverConfiguration.from_omega_conf(OmegaConf.load(config_filename))

            if len(self.rules)==0: #TODO create a rulehandler to enable different backend storage of rules. 
                for rule_key, rule_file_name in self.rule_names.items():
                    _conf = os.path.join(RULES_ROOT_PATH,self.cafe_id,RULES_PARENT_FOLDER,rule_key,rule_file_name)
                    if '.yaml' not in _conf:
                        _conf+='.yaml'
                    _rule_list = RuleList.from_omega_conf(OmegaConf.load(_conf))
                    if len(self.rules)==0:
                        self.rules = _rule_list.rules
                    else:
                        self.rules.extend(_rule_list.rules)
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
                        #event_listeners=self.streaming_event_listeners,
                        logger_level=logging.ERROR,
                        rules= self.rules,
                        stream=self.stream,
                        #tools = [WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])],
        )
        
        """
            self.agent = Agent(prompt_driver=OpenAiChatPromptDriver(model='gpt-3.5-turbo', stream=self.stream),
                            logger_level=logging.ERROR,
                            rules= self.rules,
                            stream=self.stream,)
        pass 
        """

    def run(self,*args,**kwargs):
        return self.agent.run(*args,**kwargs)
    
if __name__=='__main__':
    yak = YakAgent(
            cafe_id='Twist',
            model_driver_config_name='zephyr_7B_model_driver',
            rule_names={'restaurant':'json_menu','agent':'average_agent'},
            stream=True
        )
    from griptape.utils import Chat
    Chat(yak.agent).start()