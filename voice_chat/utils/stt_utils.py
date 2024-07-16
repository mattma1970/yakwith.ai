from typing import Dict
import logging
from yak_agents import YakServiceAgent, ExternalServiceAgent

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
                f"""Is the following SENTANCE a completed thought or not. Ask yourself the QUESTION 'Assuming I have all the relevant context, does the SENTANCE represent a completed sentance in a long conversation between a waiter in a cafe and a customer'. I'll give you 2 examples: first example, a customer says 'How much' and you say 'REPLY' because you can assume that you already have context for the item the customer is refering to; example two: a customer says 'Does the toast contain', you would reply 'WAIT' because its likely the customer is about to mention the item they are enquiring about. Your TASK is to decide if the SENTANCE represents a completed thought / question. If you decide it is complete then you must say REPLY otherwise say WAIT. If you decide to WAIT then explain the reason for your decision in less than 5 words. The SENTANCE is '{input_text}'. You can assume that you have all the context for the conversation that is needed to understand which menu items are being indirectly referred to.""",
            ]

            prompt_template = prompt_templates[2]

            prompt: str = prompt_template.replace("{input_text}", input_text)

        response: str = None
        ret: Dict = {"answer": True, "reason": "Default"}
        try:
            # use defaults set on server.
            if isinstance(service_agent, YakServiceAgent):
                response = service_agent.run(prompt)
                max_tries = 3
                while max_tries >= 0:
                    response_text = response.output_task.output.value.strip()
                    if "PAUSED" in response_text or "FINISHED" in response_text:
                        break
                    else:
                        max_tries -= 1
                if max_tries < 0:
                    raise RuntimeError(
                        "No response from the service agent for checking thought completeness."
                    )
            elif isinstance(service_agent, ExternalServiceAgent):
                if service_agent.stream is True:
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
                except Exception as e:
                    logger.error(f"Invalid json returned from completeness check: {e}")
            else:
                raise RuntimeWarning("IsCompleteThought only support JSON responses.")
        except Exception as e:
            ret = {}
            logger.error(f"Error in isCompleteThought: {e}")
        return ret
