from typing import Optional, Dict, Union, List
from os import PathLike
from pydantic import BaseModel
from dataclasses import dataclass
from attrs import define, field, Factory

from griptape.rules import Rule
from griptape.events import EventListener
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf

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

@define
class ModelDriverConfiguration:
    """
        LLM Driver Configuration
        'stream' is used both by InferenceClient (params) and by griptape as a attribute.
    """
    name: str = field()
    model: str = field()
    pretrained_tokenizer: str = field()
    token: Optional[str] = field(kw_only=True)
    # InferenceClient task specific parameters. e.g text_generation and TGI serving see huggingface_hub.inference._text_generation.TextGenerationParameters
    params: Optional[Dict] = field(kw_only=True) 
    timeout: Optional[float] = field(kw_only=True)
    headers: Optional[Dict[str, str]] = field(kw_only=True)
    cookies: Optional[Dict[str, str]] = field(kw_only=True)

    task: Optional[str] = field()
    stream: Optional[bool] = field(kw_only=True)
    stream_chunk_size: Optional[int] = field(kw_only=True)

    def __attrs_post_init__(self):
        """ Catch a mis-match in stream setting if its specified anywhere. """
        try:
            if self.stream or 'stream' in self.params:
                if self.stream != getattr(self.params(), 'stream'):
                    raise RuntimeError(f'ModelDriverConfiguration stream mismatch. params["stream"] and "stream" must be the same.')
        except:
            pass

    @classmethod
    def from_omega_conf(cls, conf: DictConfig, force_interpolation: bool = True):
        for key in conf:
            if hasattr(cls, key):
                _value = getattr(conf, key)
                if type(_value)==DictConfig:
                    _value = OmegaConf.to_container(_value, resolve=True)
                setattr(cls, key, _value)
        return cls

@define
class RuleList:
    """ Stringified rules for a restaurants and Agents. Makes it practical to save rules with OmegaConf"""
    name: Optional[str] = field(default='no_rules')
    rules: Optional[List[Rule]] = field(default=Factory(list),kw_only=True)

    @classmethod
    def from_omega_conf(cls, conf: Dict):
        for key in conf:
            if hasattr(cls, key):
                _value = getattr(conf, key)
                if key == "rules":
                    rule_list = [Rule(rule_string) for rule_string in _value]
                    setattr(cls, key, rule_list)
                else:
                    setattr(cls, key, _value)
        return cls
