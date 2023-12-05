from typing import List, Dict, Union, Tuple, Any
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from uuid import uuid4
import logging
import os
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from dataclasses import field  # pydantic.dataclasses doesn't ahve a field method

logger = logging.getLogger(__name__)

"""
    All classes related to MongoDB and pymongo.
"""


class DatabaseConfig:
    """Admin helper for mongodb resources"""

    def __init__(self, config: Dict):
        self.client = MongoClient(
            f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@{config.database.url}"
        )
        self.db = self.client[config.database.name]
        self.cafes = self.db[config.database.default_collection]


@dataclass(kw_only=True)
class Menu:
    """Menu data class for pymongo"""

    menu_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "No name set"
    raw_image_rel_path: str = ""
    raw_image_data: str = ""
    ocr_image_rel_path: str = ""
    ocr_image_data: str = ""
    menu_text: Union[Dict, str] = ""
    valid_time_range: Dict[str, datetime.datetime] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)

    def to_dict(self):
        # Convert Menu instance to dictionary
        return {
            "menu_id": self.menu_id,
            "name": self.name,
            "raw_image_rel_path": self.raw_image_rel_path,
            "raw_image_data": self.raw_image_data,
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
            name=data.get("name", ""),
            raw_image_rel_path=data.get("raw_image_rel_path", ""),
            raw_image_data=data.get("raw_image_data", ""),
            ocr_image_rel_path=data.get("ocr_image_rel_path", ""),
            ocr_image_data=data.get("ocr_image_data", ""),
            menu_text=data.get("menu_text", ""),
            valid_time_range=data.get("valid_time_range", {}),
            rules=data.get("rules", []),
        )


@dataclass(kw_only=True)
class Cafe:
    """Cafe data class for data modelling pymongo"""

    business_uid: str = ""
    default_avatar: str = ""
    house_rules: List[str] = field(default_factory=list)
    menus: List[Menu] = field(default_factory=list)

    def to_dict(self):
        # Convert Cafe instance to dictionary
        return {
            "business_uid": self.business_uid,
            "default_avatar": self.default_avatar,
            "house_rules": self.house_rules,
            "menus": [menu.to_dict() for menu in self.menus],
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Create Cafe instance from dictionary"""

        return cls(
            business_uid=data["business_uid"],
            default_avatar=data.get("default_avatar", "default"),
            house_rules=data.get("house_rules", []),
            menus=[Menu.from_dict(menu_data) for menu_data in data.get("menus", [])],
        )


class MenuModel(BaseModel):
    """Pydantic wrapper around Menu for sending back to client"""

    __root__: Menu

    class Config:
        artitrary_types_allowed = True  # skip validation
        orm_mode = (
            True  # object-relation-mapping allows python classes to be used in pydantic
        )


class Helper:
    def __init__(self):
        pass

    @classmethod
    def exists(cls, db: DatabaseConfig, business_uid: str) -> bool:
        ret: int = 0
        try:
            ret = db.cafes.countDocuments({"business_uid": business_uid})
        except Exception as e:
            logger.error(f"DB exist? error: {e}")
        return ret > 0

    @classmethod
    def get_cafe(
        cls, db: DatabaseConfig, business_uid: str, addition_criteria: Dict = None
    ) -> Cafe:
        """Find a cafe based on business_id and arbitrary, valid pymongo json object"""
        cafe: Cafe = None
        try:
            query_obj = {"business_uid": business_uid}
            if addition_criteria:
                query_obj = query_obj | addition_criteria
            cafe_dict = db.cafes.find_one(query_obj)  # Rerturns a dict
            cafe = Cafe.from_dict(cafe_dict)
        except Exception as e:
            logger.error(f"DB exist? error: {e}")
        return cafe

    @classmethod
    def get_menu_list(cls, db: DatabaseConfig, business_uid: str) -> List[Menu]:
        ret: List[Menu] = []
        try:
            cafe: Cafe = Cafe.from_dict(
                db.cafes.find_one({"business_uid": business_uid})
            )
            ret = cafe.menus
        except Exception as e:
            logger.warning(f"DB record for cafe {business_uid} not found: {e}")
        return ret

    @classmethod
    def save_menu(
        cls, db: DatabaseConfig, business_uid: str, new_menu: Menu
    ) -> Tuple[bool, str]:
        """
        Save menu to new or existing cafe.

        @return:
            ok: bool
            msg: error message if any
        """
        cafe: Cafe = cls.get_cafe(db, business_uid=business_uid)
        ok: bool = False
        msg: str = ""
        try:
            if cafe:
                cafe.menus.append(new_menu)
                db.cafes.update_one(
                    {"business_uid": business_uid}, {"$set": cafe.to_dict()}
                )
            else:
                # Create a new Cafe object and insert it into MongoDB
                new_cafe = Cafe(business_uid=business_uid, menus=[new_menu])
                db.cafes.insert_one(new_cafe.to_dict())
            ok = True
        except Exception as e:
            logger.error(f"Error saving new manu: {e}")
            msg = str(e)

        return ok, msg

    @classmethod
    def delete_one_menu(
        cls, db: DatabaseConfig, business_uid: str, menu_id: str
    ) -> Tuple[bool, str, Cafe]:
        ok: bool = False
        msg: str = ""

        try:
            cafe_dict: Dict = db.cafes.find_one(
                {"business_uid": business_uid, "menus.menu_id": menu_id}
            )
            cafe: Cafe = Cafe.from_dict(cafe_dict)
            if cafe:
                menus = [
                    menu.to_dict() for menu in cafe.menus if menu.menu_id != menu_id
                ]
                db.cafes.update_one(
                    {"business_uid": cafe.business_uid}, {"$set": {"menus": menus}}
                )
                ok = True
            else:
                logger.error(f"menu_id {menu_id} not found. Delete failed.")
                msg = f"Erorr: Failed to delete menu {str(e)}"
        except Exception as e:
            logger.error(f"Failed to delete menu with err: {str(e)}")
            ok = False
            msg = f"Erorr: Failed to delete menu {str(e)}"

        return ok, msg

    @classmethod
    def update_menu_field(
        cls,
        db: DatabaseConfig,
        business_uid: str,
        menu_id: str,
        value: Any,
        field: str = "menu_text",
    ) -> Tuple[bool, str]:
        """Update the menu.[column names] that passed in with the"""
        updated = False
        ok: bool = False
        msg: str = ""
        try:
            # Get the cafe
            cafe: Cafe = cls.get_cafe(db, business_uid, {"menus.menu_id": menu_id})
            # Update the selected field
            for menu in cafe.menus:
                if menu.menu_id == menu_id:
                    if hasattr(menu, field):
                        setattr(menu, field, value)
                        updated = True
                        break
                    else:
                        logger.error(f"Menu does not have a field called {field}")
            if updated:
                _menus = [menu.to_dict() for menu in cafe.menus]
                db.cafes.update_one(
                    {"business_uid": cafe.business_uid}, {"$set": {"menus": _menus}}
                )
            else:
                logger.info(
                    f"No update to {field} for business {business_uid}, menu_id {menu_id}"
                )
            ok = True
        except Exception as e:
            msg = f"Failed to update menu {menu_id} for business {business_uid}: {e}"
            logger.error(msg)
        return ok, msg
