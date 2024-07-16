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
from yak_agents import YakAgent
from data_classes.data_models import ModelChoice
from data_classes.mongodb import DatabaseConfig
from data_classes import MenuHelper, Cafe
from data_classes.mongodb import ModelHelper

logger = logging.getLogger(__name__)


class YakServiceAgent(YakAgent):
    """For typing purposes"""

    def __init__(self, **kwargs):
        super(YakServiceAgent, self).__init__(**kwargs)


class YakServiceAgentFactory:
    @classmethod
    def create(
        cls,
        business_uid: str,
        model_choice: ModelChoice = None,
        database: DatabaseConfig = None,
    ) -> YakServiceAgent:
        """
        Overloaded function to create a YakServiceAgent.
        If ModelChoice is not provided, then the services_model in the database is used as the service model.
        """
        service_agent: YakServiceAgent = None

        logger.info(f"Creating service agent for: {business_uid}")
        if model_choice is None and database is not None:
            # Get the locally configured Service Agent model
            cafe: Cafe = MenuHelper.get_cafe(database, business_uid)
            model_choice: ModelChoice = ModelHelper.get_model_by_id(
                database, cafe.services_model
            )  # TODO allow an alternative model to be specified.

        try:
            service_agent = YakServiceAgent(
                business_uid=business_uid,
                stream=False,
                model_choice=model_choice,
                enable_memory=False,
            )
            logger.info(f"Ok. Created service agent for llm:{model_choice.name}")
        except Exception as e:
            msg = (
                f"{__name__}: A problem occured while creating a yak service agent: {e}"
            )
            logger.error(msg)

        return service_agent

    @classmethod
    def create_from_yak_agent(cls, yak) -> YakServiceAgent:
        """Create a service agent using the same model (and its settings) as used by the yak conversation agent."""
        service_agent: YakServiceAgent = None
        try:
            service_agent = YakServiceAgentFactory.create(
                yak.business_uid, yak.model_choice
            )
        except Exception as e:
            msg = f"A problem occured while creating a yak_agent: {e}"
            logger.error(msg)

        return service_agent
