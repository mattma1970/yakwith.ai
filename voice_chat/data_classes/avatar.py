from typing import Dict
from attr import define, Factory, field
import logging
from voice_chat.text_to_speech.TTS_enums import VisemeSource

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
            except Exception as e:
                logger.error(
                    f"Invalid visemesource set in avatar configuration. Must be api or local. Defaulting to api: {e}"
                )
                self.viseme_source = VisemeSource.API
