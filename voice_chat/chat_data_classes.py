from typing import Optional, Dict
from pydantic import BaseModel

class ApiUserMessage(BaseModel):
    """Message sent from App"""
    user_input: str
    session_id: str
    user_id:Optional[str]

class AppParameters(BaseModel):
    """Parameters app"""
    name: str
    action: str
    params: Dict[str,str]
