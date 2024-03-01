# (c) yakwith.ai

"""
YakServiceAgents are YakAgents that are restricted for performing low level tasks. e.g assess if an utterance is completed.
YakServiceAgents:
    - have access to settings for the current business (business_uid)
    - do not use conversation memory
    - do not stream responses.

YakServiceAgents are intended to be cached (service_agent_registry) rather than recreated on every invocation and so this class should not be instatiated. 

"""

import logging
from voice_chat.yak_agents import YakAgent
from voice_chat.data_classes.data_models import ModelChoice

logger = logging.getLogger(__name__)


class YakServiceAgent(YakAgent):
    """For typing purposes"""

    def __init__(self, **kwargs):
        super(YakServiceAgent, self).__init__(**kwargs)


class YakServiceAgentFactory:
    @classmethod
    def create(cls, business_uid: str, model_choice: ModelChoice) -> YakServiceAgent:
        service_agent: YakServiceAgent = None

        logger.info(f"Creating service agent for: {business_uid}")
        try:
            service_agent = YakServiceAgent(
                business_uid=business_uid,
                stream=False,
                model_choice=model_choice,
                enable_memory=False,
            )
            logger.info(f"Ok. Created service agent for llm:{model_choice.name}")
            ok = True
        except Exception as e:
            msg = (
                f"{__name__}: A problem occured while creating a yak service agent: {e}"
            )
            logger.error(msg)

        return service_agent

    @classmethod
    def create_from_yak_agent(cls, yak) -> YakServiceAgent:
        service_agent: YakServiceAgent = None
        try:
            service_agent = YakServiceAgentFactory.create(
                yak.business_uid, yak.model_choice
            )
            ok = True
        except Exception as e:
            msg = f"A problem occured while creating a yak_agent: {e}"
            logger.error(msg)

        return service_agent
