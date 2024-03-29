"""
An agent for miscellaneous ai-powered functions in the app. For example, correcting ORC text which is inherently noisey. 
"""

from typing import List, Optional, Tuple, Union, Generator
from dotenv import load_dotenv
from attrs import define, field, Factory, validators
import os
from enum import Enum
import logging

from griptape.structures import Agent
from griptape.utils import Chat, PromptStack
from griptape.drivers import (
    # HuggingFaceInferenceClientPromptDriver,
    HuggingFaceHubPromptDriver,
    OpenAiChatPromptDriver,
)
from griptape.utils.stream import Stream


logger = logging.getLogger(__name__)
load_dotenv()

Task = Enum("Task", ["NONE", "TEXT_CORRECTION"])


class Provider(Enum):
    OPENAI = 0
    LOCAL = 1


def to_provider(value):
    if isinstance(value, str):
        try:
            return Provider[value]
        except KeyError:
            logger.error(f"'{value}' is not a valid option for Provider")
            return Task.NONE
    return value


@define
class ExternalServiceAgent:
    """
    Misc 3rd-party-LLM-powered non-conversation functions such as processing and cleaning OCR data.
    Supports streaming

    @args:
        provider: Provider: the LLM API service provider: ;passed in as a string and converted to Provider enum.
        model: str: name of LLM to use from provider
        task: str: NOT YET IMPLEMENTED.
        agent: griptape.structure.agent
        stream: bool: whether the response should be streamed back of sent on completion.
    @returns:
        generator if stream==true
        LLM text response if not streaming != True
    """

    provider: Provider = field(
        converter=to_provider, default=os.environ["SERVICE_AGENT_PROVIDER"]
    )
    model: Optional[str] = field(default=os.environ["SERVICE_AGENT_MODEL"])
    task: Optional[Task] = field(default=Task.NONE)
    agent: Agent = field(default=None)
    stream: bool = field(default=False)

    def __attrs_post_init__(self):
        try:
            if self.provider == Provider.OPENAI:
                self.agent = Agent(
                    memory=None,
                    prompt_driver=OpenAiChatPromptDriver(
                        temperature=0.4,
                        model=self.model,
                        api_key=os.environ["OPENAI_API_KEY"],
                        stream=self.stream,
                    ),
                    logger_level=logging.ERROR,
                    stream=self.stream,
                )
        except Exception as e:
            logger.error(f"Failed to create LLM driver for OPENAI: {e}")

    def response_generator(self, prompt: str) -> Generator[str, None, None]:
        """A generator with the LLM response."""
        stream = Stream(self.agent).run(prompt)
        for chunk in stream:
            yield (chunk.value)

    def do_job(self, prompt: str):
        if self.agent is not None:
            if self.agent.stream:
                return self.response_generator(prompt=prompt)
            else:
                # text_response = self.agent.run(prompt).output_task.output.to_text()  ### Use when upgrading.
                text_response = self.agent.run(prompt).output.to_text()
                return text_response
        else:
            raise RuntimeError(f"Error running non stream service agent job.")


if __name__ == "__main__":
    agent = ExternalServiceAgent(provider="OPENAI")
    print(agent.do_job("How many horns are there in a herd of 7 unicorns?"))
