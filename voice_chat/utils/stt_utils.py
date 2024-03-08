import math
import re, json
from typing import Dict, List, Any, Union, Tuple
import logging
import io, os
from voice_chat.yak_agents.external_service_agent import Task
from voice_chat.yak_agents import YakServiceAgent, ExternalServiceAgent
import logging

logger = logging.getLogger(__name__)


class STTUtilities:
    @classmethod
    def isCompleteThought(
        cls,
        input_text: str,
        prompt_template: str = "",
        getJSON=True,
        service_agent: YakServiceAgent = None,
    ):
        """
        Use the current LLM to checvk if the text that was sent was a complete thought and hence should be replied to with a direct response.
        The prompt template should contain a placeholder {text} for the input text to be tested for completeness.
        @args:
            prompt_template: str: prompt with placeholder for text. If empty, the default prompt template below is used.
            getJSON: bool: the response from the model is assumed to be json.
        @returns:
            Dict: 'answer':<bool>,'reason': rationale for the answer.

        Note: from early experiments, requiring a rationale for the answer leads to materially better results than just asking for a yes not response.
        """
        if prompt_template.strip() == "":
            prompt_template = f"""You are a waiter at a restaurant. Does the following statement make sense to you, ignore spelling mistakes 
                                and things that might be food or beverage items even if its not a common item:\n
                                '{input_text}'?\n
                                First answer with 'yes' or 'no' only and then give an explanation of your answer in less than 6 words"""

            prompt: str = prompt_template.replace("{input_text}", input_text)

        response: str = None
        ret: Dict = {"answer": True, "reason": "Default"}
        try:
            # use defaults set on server.
            if isinstance(service_agent, YakServiceAgent):
                response = service_agent.run(prompt)
                response_text = response.output.value.strip()
            elif isinstance(service_agent, ExternalServiceAgent):
                if service_agent.stream == True:
                    raise RuntimeError(
                        "Streaming external service agents are not supported by isCompleteThought"
                    )
                response_text = service_agent.do_job(prompt)
            else:
                logger.error(
                    f"No service agent found or service agent type {type(service_agent)} isn"
                    "t supported for isCompleteThought."
                )
            if getJSON:
                try:
                    ret["answer"] = False if "no" in response_text[:3].lower() else True
                    ret["reason"] = response.output.value
                except:
                    logger.error("Invalid json returned from completeness check")
            else:
                raise RuntimeWarning("IsCompleteThought only support JSON responses.")
        except Exception as e:
            ret = {}
            logger.error(f"Error in isCompleteThought: {e}")
        return ret
