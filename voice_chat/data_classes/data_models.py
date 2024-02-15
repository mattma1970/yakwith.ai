from typing import List, Dict, Union, Tuple, Any, Optional
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from uuid import uuid4
import logging
import os
import json
from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass
from dataclasses import field  # pydantic.dataclasses doesn't ahve a field method
from enum import Enum

logger = logging.getLogger(__name__)

"""
Pydantic.dataclasses - Data models for DB - thses classes are not database specific
Note images are stored outside the database and the data is inserted JIT as needed.

"""


class ImageSelector(Enum):
    RAW = 0
    THUMBNAIL = 1
    OCR = 2


@dataclass(kw_only=True)
class Menu:
    """
    A single image menu. 'collection' is used to group multiple pages
    into a single menu. The Menu with collection.sequence_number == 1 is used to store collected text for all pages.
    """

    menu_id: str = field(default_factory=lambda: str(uuid4()))
    collection: Optional[Dict[str, str]] = field(
        default_factory=lambda: {"grp_id": "", "sequence_number": 0}
    )  # For grouping multi-image menus
    name: str = "No name set"
    raw_image_rel_path: Optional[str] = ""
    raw_image_data: Optional[str] = ""
    thumbnail_image_rel_path: Optional[str] = ""
    thumbnail_image_data: Optional[str] = ""
    ocr_image_rel_path: Optional[str] = ""
    ocr_image_data: Optional[str] = ""
    menu_text: str = ""
    valid_time_range: Dict[str, Optional[datetime]] = field(default_factory=dict)
    rules: str = ""

    @validator("valid_time_range", pre=True, each_item=True)
    def set_utc_timezone(cls, v):
        """Force datetime to be timezone aware and use UTC if no timezone is specified. TZ info is lost when sending from front end."""
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def to_dict(self):
        # Convert Menu instance to dictionary
        return {
            "menu_id": self.menu_id,
            "collection": self.collection,
            "name": self.name,
            "raw_image_rel_path": self.raw_image_rel_path,
            "raw_image_data": self.raw_image_data,
            "thumbnail_image_rel_path": self.thumbnail_image_rel_path,
            "thumbnail_image_data": self.thumbnail_image_data,
            "ocr_image_rel_path": self.ocr_image_rel_path,
            "ocr_image_data": self.ocr_image_data,
            "menu_text": self.menu_text,
            "valid_time_range": self.valid_time_range,
            "rules": self.rules,
        }

    @classmethod
    def from_dict(cls, data):
        # Create Menu instance from dictionary
        return cls(
            menu_id=data.get("menu_id"),
            collection=data.get("collection", {}),
            name=data.get("name", ""),
            raw_image_rel_path=data.get("raw_image_rel_path", ""),
            raw_image_data=data.get("raw_image_data", ""),
            thumbnail_image_rel_path=data.get("thumbnail_image_rel_path"),
            thumbnail_image_data=data.get("thumbnail_image_data"),
            ocr_image_rel_path=data.get("ocr_image_rel_path", ""),
            ocr_image_data=data.get("ocr_image_data", ""),
            menu_text=data.get("menu_text", ""),
            valid_time_range=data.get(
                "valid_time_range", {}
            ),  # stores dates as ISODate. When retrieve they are automatically converted to datetime.datetime by pymongo
            rules=data.get("rules", ""),
        )


@dataclass(kw_only=True)
class Cafe:
    """Cafe data class for data modelling pymongo"""

    business_uid: str = ""
    default_avatar: Optional[str] = ""
    avatar_settings: Optional[Dict] = field(
        default_factory=dict
    )  # Make this a dict for future expansion
    house_rules: Optional[str] = ""
    menus: Optional[List[Menu]] = field(default_factory=list)
    notes: Optional[str] = ""
    model: str = "none"

    def to_dict(self):
        # Convert Cafe instance to dictionary
        return {
            "business_uid": self.business_uid,
            "default_avatar": self.default_avatar,
            "avatar_settings": self.avatar_settings,
            "house_rules": self.house_rules,
            "menus": [menu.to_dict() for menu in self.menus],
            "notes": self.notes,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Create Cafe instance from dictionary"""

        return cls(
            business_uid=data["business_uid"],
            default_avatar=data.get("default_avatar", "default"),
            avatar_settings=data.get("avatar_settings", {}),
            house_rules=data.get("house_rules", ""),
            menus=[Menu.from_dict(menu_data) for menu_data in data.get("menus", [])],
            notes=data.get("notes", ""),
            model=data.get("model"),
        )


@dataclass(kw_only=True)
class ModelChoice:
    """LLM model spec for model collection"""
    id: str = field(default ="")
    name: str = field(default="")
    config: List[Union[Dict]] = field(default_factory=list)
    provider: str = field(default="")
    driver_file_name: Optional[str] = field(default="")

    @classmethod
    def from_dict(cls, data: Dict):
        a: ModelChoice = cls(
            id = data["id"],
            name = data.get("name",""),
            config = [_config for _config in data.get("config",[])],
            provider = data.get("provider",""),
            driver_file_name = data.get("driver_file_name",""),
        )
        return a
        
    @classmethod
    def to_dict(self):
        return {
                "id": self.id,
                "name": self.name,
                "config": self.config,
                "provider": self.provider
                }