from typing import Optional, Dict
from pydantic import BaseModel


""" Pydantic classes """


class ApiUserMessage(BaseModel):
    """Message sent from App"""

    user_input: str
    session_id: str
    user_id: Optional[str]


class AppParameters(BaseModel):
    """Parameters app"""

    name: str
    action: str
    params: Dict[str, str]


class SessionStart(BaseModel):
    """Message to create a new chat bot"""

    session_id: str
    business_uid: str
    menu_id: str
    avatar_personality: Optional[
        str
    ]  # free text describing the avatars personality and behaviour.
    stream: bool  # whether responses should be streamed back.
    user_id: Optional[str] = None


class ServiceAgentRequest(BaseModel):
    """Request object for agent for non-avatar, service/utiliity functions such as checking completeness of utterance or parsing to JSON."""

    session_id: str
    business_uid: str
    provider: str = "LOCAL"


class SttTokenRequest(BaseModel):
    """Request for temporary token for stt service"""

    service_name: str
    client_authorization_key: str


class ThirdPartyServiceAgentRequest(BaseModel):
    """Getting a one time response from LLM API provider."""

    task: Optional[str]
    prompt: str
    stream: bool


class LocalServiceAgentResquest(BaseModel):
    prompt: str
    session_id: str
    business_uid: str
