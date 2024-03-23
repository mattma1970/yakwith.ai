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
            prompt_templates = [
                f"""You are a waiter at a restaurant. Does the following statement make sense to you, 
                                ignore spelling mistakes and things that might be food or beverage items even if its not a common item:\n'{input_text}'?\nFirst answer with 'yes' or 'no' only and then give 
                              an explanation of your answer in less than 6 words""",
                f"""You are a highly intuitive waiter at a cafe. When ordering, sometimes customers start speaking and pause while they think. A customer at a cafe SAID '{input_text}'.\n You must decide if have PAUSED to think or they have FINISHED. Based on what the customer SAID, do you think they have PAUSED or FINISHED.""",
            ]

            prompt_template = prompt_templates[1]

            prompt: str = prompt_template.replace("{input_text}", input_text)

        response: str = None
        ret: Dict = {"answer": True, "reason": "Default"}
        try:
            # use defaults set on server.
            if isinstance(service_agent, YakServiceAgent):
                response = service_agent.run(prompt)
                max_tries = 3
                while max_tries >= 0:
                    response_text = response.output.value.strip()
                    if "PAUSED" in response_text or "FINISHED" in response_text:
                        break
                    else:
                        max_tries -= 1
                if max_tries < 0:
                    raise RuntimeError(
                        "No response from the service agent for checking thought completeness."
                    )
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
                    ret["answer"] = False if "PAUSED" in response_text else True
                    ret["reason"] = response_text
                except:
                    logger.error("Invalid json returned from completeness check")
            else:
                raise RuntimeWarning("IsCompleteThought only support JSON responses.")
        except Exception as e:
            ret = {}
            logger.error(f"Error in isCompleteThought: {e}")
        return ret
