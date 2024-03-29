from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Callable, Any
from dotenv import load_dotenv
from attrs import define, field, Factory, validators
from queue import Queue
from enum import Enum

from griptape.structures import Agent
from griptape.memory.structure import ConversationMemory
from griptape.utils import Chat, PromptStack
from griptape.drivers import (
    HuggingFaceHubPromptDriver,
    OpenAiChatPromptDriver,
)
from griptape.tokenizers import HuggingFaceTokenizer
from griptape.events import CompletionChunkEvent, FinishStructureRunEvent, EventListener
from griptape.rules import Rule, Ruleset

from transformers import AutoTokenizer
import os
import json
import logging
from omegaconf import OmegaConf

from voice_chat.data_classes import ModelDriverConfiguration, RuleList
from voice_chat.data_classes import PromptBuffer
from voice_chat.data_classes.mongodb import ModelHelper, ModelChoice
from voice_chat.text_to_speech.TTS import TextToSpeechClass

logger = logging.getLogger(__name__)

load_dotenv()


class YakStatus(Enum):
    """Status values of the agent that can be used as a session state."""

    INITIALIZING = 0
    IDLE = 1
    GENERATING_TEXT = 2
    TALKING = 3


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

        rules : List[str]: List of natural language prompts to be permanently saved to the LLM context (survives LLM context pruning in Griptape)

        user_id: a uid for the user if provided. Reserved for personalisation feature in the future.
        task: the function name of the InferenceClient task that will be served.
        stream: streaming flag for both the prompt driver and the Agent.

        :: Attributes not initialized ::
        streaming_event_listeners: List[EventListener], optional: List of event listeners for the griptape.agent. Mostly useful for debugging.
        output_fn: Callable, optional: output of streaming responses.

        agent: Agent : The griptape.agent

    """

    business_uid: str = field(default="default", kw_only=True)

    """ Model and prompting properties """
    model_choice: ModelChoice = field(default=None)  # User choice of model in settings.
    model_driver_config: Optional[ModelDriverConfiguration] = field(
        default=None
    )  # Allow config to be injected.

    rule_names: Optional[Dict] = field(
        default=Factory(dict)
    )  # TODO remove. deprecated after adding mongo backend
    rules: Optional[list[str]] = field(default=Factory(list))
    user_id: Optional[str] = field(default=None)
    prompt_buffer: PromptBuffer = field(
        factory=PromptBuffer
    )  # Used to collect incomplete prompts until a complete one is ready.

    """ Properties related to the griptape Agent """
    task: Optional[str] = field(default="text_generation", kw_only=True)
    enable_memory: bool = field(
        default=True
    )  # Uses Griptape.ConversationMemory as default of False.
    stream: Optional[bool] = field(default=False)
    streaming_event_listeners: Optional[List[EventListener]] = field(
        default=None, kw_only=True
    )
    output_fn: Optional[Callable] = field(init=False)
    agent: Agent = field(init=False)
    agent_status: YakStatus = field(
        default=YakStatus.IDLE
    )  # Access via status property

    """ Avatar Properties """
    voice_id: str = field(default="NONE")  # Agent voice id from STT provider.
    avatar_config: Dict = field(
        default=Factory(dict)
    )  # Primarily for avatar animations including blendshapes.
    TextToSpeech: TextToSpeechClass = field(
        default=None
    )  # Object dealing with speech synthesis, visemes and blendshapes. Assinged on first use.

    notes: str = field(default="")  # Misc text field with unspecified primary use case.

    def __attrs_post_init__(self):
        try:
            if self.model_driver_config is None:
                if self.model_choice.driver_file_name == "":
                    logger.error("No model driver specified")
                else:
                    # model driver config not injected and is spefied in the model_choice config.
                    if ".yaml" not in self.model_choice.driver_file_name:
                        self.model_choice.driver_file_name += ".yaml"
                    config_filename = os.path.join(
                        f"{os.environ['APPLICATION_ROOT_FOLDER']}/{os.environ['MODEL_DRIVER_CONFIG_PATH']}",
                        self.model_choice.driver_file_name,
                    )
                    self.model_driver_config = ModelDriverConfiguration.from_omega_conf(
                        OmegaConf.load(config_filename)
                    )

        except Exception as e:
            raise RuntimeError(f"Error loading agent related config file: {e}")

        if self.stream:
            self.streaming_event_listeners = [
                EventListener(
                    lambda x: print(x.token, end=""), event_types=[CompletionChunkEvent]
                ),
                EventListener(
                    lambda _: print("\n"), event_types=[FinishStructureRunEvent]
                ),
            ]
            self.output_fn = lambda x: x
        else:
            self.streaming_event_listeners = []
            self.output_fn = lambda x: print(x)

        if self.model_choice.provider == "tgi.local":
            # Remove max_new_tokens and stream from params to avoid duplicate parameters in text_generation.
            max_new_tokens = self.model_driver_config.params.pop("max_new_tokens", 250)
            do_stream = self.model_driver_config.params.pop("stream", True)
            self.agent = Agent(
                prompt_driver=HuggingFaceHubPromptDriver(
                    api_token=self.model_driver_config.token,
                    model=self.model_driver_config.model,
                    tokenizer=HuggingFaceTokenizer(
                        tokenizer=AutoTokenizer.from_pretrained(
                            os.path.join(
                                os.environ["APPLICATION_ROOT_FOLDER"],
                                self.model_driver_config.pretrained_tokenizer,
                            )
                        ),
                        max_output_tokens=max_new_tokens,
                    ),
                    params=self.model_driver_config.params,
                ),
                # event_listeners=self.streaming_event_listeners,
                logger_level=logging.ERROR,
                rules=[Rule(rule) for rule in self.rules],
                stream=self.stream,
                # tools = [WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])],
            )
            # Turn off conversation memory if disabled in yak_agent. This is done outside the Agent creation
            # in order that the conversation memory default can access gripetape.ai configuration setting
            if self.enable_memory is False:
                self.agent.conversation_memory = None
        elif self.model_choice.provider == "openai":
            try:
                self.agent = Agent(
                    prompt_driver=OpenAiChatPromptDriver(
                        model=self.model_choice.name, stream=self.stream
                    ),
                    logger_level=logging.ERROR,
                    rules=[Rule(rule) for rule in self.rules],
                    stream=self.stream,
                    memory=ConversationMemory() if self.enable_memory else None,
                )
            except Exception as e:
                logger.error(
                    f"""Failed to create Agent with model {self.model_choice.name} 
                             for provider {self.model_choice.provider}. Please check model name, 
                             provider in app user settings.:x"""
                )
        else:
            logger.error("ModelChoice provider is not recognised.")
        self.status = YakStatus.IDLE

    @property
    def status(self) -> YakStatus:
        """Functions as a shared state for api functions."""
        return self.agent_status

    @status.setter
    def status(self, value: YakStatus):
        self.agent_status = value

    def run(self, *args, **kwargs):
        return self.agent.run(*args, **kwargs)

    @property
    def usingCache(self) -> bool:
        """usecache is a setting in avatar settings json."""
        return "usecache" in self.avatar_config and self.avatar_config["usecache"]


if __name__ == "__main__":
    """yak = YakAgent(
        cafe_id="Twist",
        model_driver_config_name="zephyr_7B_model_driver",
        rule_names={"restaurant": "json_menu", "agent": "average_agent"},
        stream=True,
    )
    from griptape.utils import Chat

    Chat(yak.agent).start()"""
    pass
