from typing import Optional, Dict, Union, List, Any
from os import PathLike
import json
import re
from attr import define, Factory, field
import logging


logger = logging.getLogger(__name__)

"""
Helper functions for parsing the Avatar config json string.
Acts as a facade class for config while allowing arbitrary keys to also be added
for future extensibility
"""


@define
class AvatarConfigParser:
    config: Dict = field(default=Factory(Dict))
    blendshapes: bool = field(
        kw_only=True, default=False
    )  # Whether blendshapes should be returned for TTS
    avatar: str = field(kw_only=True, default="")  # Avatar name

    def __attrs_post_init__(self):
        if "useblendshapes" in self.config:
            self.blendshapes = self.config["useblendshapes"]
        if "avatar" in self.config:
            self.avatar = self.config["avatar"]
