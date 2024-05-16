from typing import Optional, Dict, Union, List, Any
from os import PathLike
import json
import re
from attr import define, Factory, field
import logging
from voice_chat.text_to_speech.classes.text_to_speech_enums import VisemeSource

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
    viseme_source: VisemeSource = field(default=VisemeSource.API)

    def __attrs_post_init__(self):
        if "useblendshapes" in self.config:
            self.blendshapes = self.config["useblendshapes"]
        if "avatar" in self.config:
            self.avatar = self.config["avatar"]
        if "visemesource" in self.config:
            try:
                self.viseme_source = VisemeSource[self.config["visemesource"].upper()]
            except:
                logger.error(
                    "Invalid visemesource set in avatar configuration. Must be api or local. Defaulting to api"
                )
                self.viseme_source = VisemeSource.API
