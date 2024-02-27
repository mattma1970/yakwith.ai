import math
import re, json
from typing import Dict, List, Any, Union, Tuple
import logging
import io, os
from voice_chat.yak_agents.service_agent import Task
from voice_chat.yak_agents import YakAgent
import logging

logger = logging.getLogger(__name__)


class STTUtilities:
    @classmethod
    def isCompleteThought(
        cls,
        input_text: str,
        prompt_template: str = "",
        getJSON=True,
        service_agent: YakAgent = None,
    ):
        """
        Use the current LLM to checvk if the text that was sent was a complete thought and hence should be replied to with a direct response.
        The prompt template should contain a placeholder {text} for the input text to be tested for completeness.
        @args:
            prompt_template: str: prompt with placeholder for text. If empty, the default prompt template below is used.
            getJSON: bool: the response from the model is assumed to be json.
        @returns:
            answer, possible in json format.

        Note: from early experiments, requiring a rationale for the answer leads to materially better results than just asking for a yes not response.
        """
        if prompt_template.strip() == "":
            prompt_template = """Does the Statement, quoted below, makes sense if it were spoken to a restaurant waiter. You must ignore spelling mistakes and words that could be a food item.
                                    Answer in json format with the following schema {{'answer': a boolean indicating whether the sentance makes sense, 'reason': a 6 word explaination of your reasoning}}.  Statement:'{input_text}'"""

            prompt: str = prompt_template.replace("{input_text}", input_text)

        response: str = None
        ret: Dict = {}
        try:
            # use defaults set on server.
            response = service_agent.run(prompt)
        except Exception as e:
            ret = {}
            logger.error(f"Error in isCompleteThought: {e}")
        if getJSON:
            try:
                ret = json.loads(response.output.value)
            except:
                logger.error("Invalid json returned from completeness check")
        else:
            raise RuntimeWarning("IsCompleteThought only support JSON responses.")
        return ret
