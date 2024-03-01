import json
from typing import List, Dict, Tuple, Union
from voice_chat.yak_agents import ExternalServiceAgent, YakServiceAgent
from voice_chat.utils.stt_utils import STTUtilities
from voice_chat.data_classes import PromptBuffer
import logging

from voice_chat.configs.AppConfig import Configurations

logger = logging.getLogger("YakChatAPI")


class PromptManager:
    @classmethod
    def SmartAccumulator(
        cls,
        prompt_fragment: str,
        *,
        prompt_buffer: PromptBuffer = None,
        service_agent: Union[ExternalServiceAgent, YakServiceAgent] = None,
        session_id: str = "",
    ) -> Tuple[bool, str]:
        """
        Collects prompt fragments until a coherent whole prompt is available.
        """
        rolling_prompt: str = ""
        isCompleteUtterance: bool = True
        completion_check: Dict = {}
        try:
            prompt_buffer.push(prompt_fragment)
            rolling_prompt = prompt_buffer.prompt
            if len(prompt_buffer) < Configurations.nlp.max_length_of_prompt_buffer:
                completion_check: Dict = STTUtilities.isCompleteThought(
                    input_text=rolling_prompt, service_agent=service_agent
                )
                isCompleteUtterance = completion_check["answer"]
                if isCompleteUtterance:
                    prompt_buffer.reset()
                else:
                    isCompleteUtterance = False
            else:
                # Flush the prompt buffer.
                isCompleteUtterance = True
                completion_check = {"answer": True, "reason": "Pormpt flushed"}
                prompt_buffer.reset()

        except Exception as e:
            logger.error(f"'SmartAccumulator': session_id: {session_id}: {e}")
            isCompleteUtterance = True  # If there's a problem processing things, then send it to the LLM anyhow and get it to deal with it.
        logger.debug(
            f"CompletionCheckResults:{rolling_prompt}:{json.dumps(completion_check)}"
        )
        return isCompleteUtterance, rolling_prompt
