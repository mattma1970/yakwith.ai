from attrs import define, field, Factory
import os
import glob
from os import PathLike
from typing import Optional, List, Dict, Union
from data_classes.chat_data_classes import Menu, MenuList
from datetime import datetime
from voice_chat.menus import BaseMenuHandler


@define
class MenuManager:
    """
    Class to handle all Menu operations
    """

    cafe_id: str = field()
    menu_handler: BaseMenuHandler = field()
