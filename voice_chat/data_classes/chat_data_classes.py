from typing import Optional, Dict, Union, List
from os import PathLike
from pydantic import BaseModel
from dataclasses import dataclass
from attrs import define, field, Factory, validators
from uuid import uuid4

import datetime
import pytz

import re

from griptape.rules import Rule
from griptape.events import EventListener
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf

from fastapi import UploadFile


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
    """Message to create a new session and agent"""

    cafe_id: str
    rule_names: Dict[str, str]
    stream: bool
    user_id: Optional[str] = None


class SttTokenRequest(BaseModel):
    """Request for temporary token for stt service"""

    service_name: str
    client_authorization_key: str


class ServiceAgentRequest(BaseModel):
    """Getting a one time response from LLM API provider."""

    task: Optional[str]
    prompt: str
    stream: bool


@define
class ModelDriverConfiguration:
    """
    LLM Driver Configuration targeting HF InferenceClient
    'stream' is used both by InferenceClient (params) and by griptape as a attribute.
    """

    name: str = field(default=None)
    model: str = field(default=None)
    pretrained_tokenizer: str = field(default=None)
    token: Optional[str] = field(default=None, kw_only=True)
    # InferenceClient task specific parameters. e.g text_generation and TGI serving see huggingface_hub.inference._text_generation.TextGenerationParameters
    params: Optional[Dict] = field(default=None, kw_only=True)
    timeout: Optional[float] = field(default=None, kw_only=True)
    headers: Optional[Dict[str, str]] = field(default=None, kw_only=True)
    cookies: Optional[Dict[str, str]] = field(default=None, kw_only=True)

    task: str = field(default=None, kw_only=True)
    stream: Optional[bool] = field(default=None, kw_only=True)
    stream_chunk_size: Optional[int] = field(default=None, kw_only=True)

    def __attrs_post_init__(self):
        """Catch a mis-match in stream setting if its specified anywhere."""
        try:
            if self.stream or "stream" in self.params:
                if self.stream != getattr(self.params(), "stream"):
                    raise RuntimeError(
                        f'ModelDriverConfiguration stream mismatch. params["stream"] and "stream" must be the same.'
                    )
        except:
            pass

    @classmethod
    def from_omega_conf(cls, conf: DictConfig, force_interpolation: bool = True):
        new_model_driver = ModelDriverConfiguration()
        for key in conf:
            if hasattr(new_model_driver, key):
                _value = getattr(conf, key)
                if type(_value) == DictConfig:
                    _value = OmegaConf.to_container(_value, resolve=True)
                setattr(new_model_driver, key, _value)
        new_model_driver.__attrs_post_init__()
        return new_model_driver


@define
class RuleList:
    """Convenience class for loading list[Rules] saved with OmegaConf which can only save strings."""

    name: Optional[str] = field(default="yak_agent_rules", kw_only=True)
    rules: Optional[List[Rule]] = field(default=Factory(list), kw_only=True)

    @classmethod
    def from_omega_conf(cls, conf: Dict) -> "RuleList":
        new_rule_list = RuleList()
        for key in conf:
            if hasattr(new_rule_list, key):
                _value = getattr(conf, key)
                if key == "rules":
                    rule_list = [Rule(rule_string) for rule_string in _value]
                    setattr(new_rule_list, key, rule_list)
                else:
                    setattr(new_rule_list, key, _value)
        return new_rule_list


@define
class Menu:
    """Cafe menu with date and time validity"""

    def date_check(inst, att, value):
        res = re.findall(r"\d{4}-\d{2}-\d{2}", value)
        if not (len(res) == 1 and res[0] == value):
            raise ValueError(f"{att} format for menu is not in form YYYY-mm-dd")

    def time_check(inst, att, value):
        if value is not None:
            res = re.findall(r"\d{2}:\d{2}", value)
            if not (len(res) == 1 and res[0] == value):
                raise ValueError(f"{att} format for menu is not in form HH:MM")

    cafe_id: str = field(default=None)
    menu: str = field(default=None)
    name: str = field(default=Factory(lambda: str(uuid4())), kw_only=True)
    type: str = field(default="all", kw_only=True)
    time_zone: str = field(default="Australia/Sydney", kw_only=True)
    start_date: Optional[str] = field(
        default="1900-01-01", validator=[date_check], kw_only=True
    )  # %Y-%m-%d
    end_date: Optional[str] = field(
        default="3000-01-01", validator=[date_check], kw_only=True
    )
    start_time_of_day: Optional[str] = field(
        default="00:00", validator=[time_check], kw_only=True
    )  # %H:%M
    end_time_of_day: Optional[str] = field(
        default="23:59", validator=[time_check], kw_only=True
    )

    @classmethod
    def from_omega_conf(cls, conf: DictConfig):
        new_menu = Menu()
        for key in conf:
            if hasattr(new_menu, key):
                _value = getattr(conf, key)
                setattr(new_menu, key, _value)
        return new_menu


@define
class MenuList:
    """List of all menus for a given cafe"""

    cafe_id: str = field()
    menus: List[Menu] = field(default=Factory(list))

    def __attrs_post_init__(self):
        menu_names = []
        for menu in self.menus:
            if menu.name in menu_names:
                raise ValueError(
                    f'Menu names must be unique. "{menu.name}" was repeated for establishment = {self.cafe_id}'
                )
            else:
                menu_names.append(menu.name)

    @classmethod
    def from_omega_conf(cls, conf: DictConfig):
        new_menu_list = MenuList(None)
        for key in conf:
            _value = getattr(conf, key)
            if key == "menus":
                _menus = []
                for menu in _value:
                    _menus.append(Menu(**menu))
                setattr(new_menu_list, key, _menus)
            else:
                setattr(new_menu_list, key, _value)
        new_menu_list.__attrs_post_init__()
        return new_menu_list
