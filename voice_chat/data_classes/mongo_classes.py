from attr import define, field, Factory
from typing import List, Dict, Union
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from uuid import uuid4
import logging
import os

logger= logging.getLogger(__name__)

"""
    All classes related to MongoDB and pymongo.
"""

class DatabaseConfig:
    """ Admin helper for mongodb resources """
    def __init__(self, config: Dict):
        self.client = MongoClient(f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@{config.database.url}")
        self.db = self.client[config.database.name]
        self.cafes = self.db[config.database.default_collection]

@define(kw_only=True)
class Menu:
    """Menu class for pymongo"""

    menu_id: str = field(default = Factory(lambda x: str(uuid4())))
    raw_image_rel_path: str = field(default="")
    ocr_image_rel_path: str = field(default="")  # Image with text highlighted
    menu_text: Union[Dict, str] = field(default='')
    rules: List[str] = field(default=Factory(list))

    def to_dict(self):
        # Convert Menu instance to dictionary
        return {
            "menu_id": self.menu_id,
            "raw_image_rel_path": self.raw_image_rel_path,
            "ocr_image_rel_path": self.ocr_image_rel_path,
            "menu_text": self.menu_text,
            "rules": self.rules,
        }

    @classmethod
    def from_dict(cls, data):
        # Create Menu instance from dictionary
        return cls(
            menu_id=data.get("menu_id"),
            raw_image_rel_path=data.get("raw_image_rel_path", ""),
            ocr_image_rel_path=data.get("ocr_image_rel_path", ""),
            menu_text=data.get("menu_text", ""),
            rules=data.get("rules", []),
        )


@define(kw_only=True)
class Cafe:
    """Cafe class for data modelling pymongo"""

    business_uid: str = field()
    default_avatar: str = field(default="default")
    house_rules: List[str] = field(default=Factory(list))
    menus: List[Menu] = field(default=Factory(list))
    valid_time_range: Dict[str, datetime.datetime] = field(default=Factory(dict))

    def to_dict(self):
        # Convert Cafe instance to dictionary
        return {
            "business_uid": self.business_uid,
            "default_avatar": self.default_avatar,
            "house_rules": self.house_rules,
            "menus": [menu.to_dict() for menu in self.menus],
            "valid_time_range": self.valid_time_range,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """ Create Cafe instance from dictionary """

        return cls(
            business_uid=data["business_uid"],
            default_avatar=data.get("default_avatar", "default"),
            house_rules=data.get("house_rules", []),
            menus=[Menu.from_dict(menu_data) for menu_data in data.get("menus", [])],
            valid_time_range=data.get("valid_time_range", {}),
        )


class Ops:
    def __init__(self):
        pass

    @classmethod
    def exists(cls, db: DatabaseConfig, business_uid: str) -> bool:
        ret:int = 0
        try:
            ret =  db.cafes.countDocuments({"business_uid": business_uid})
        except Exception as e:
            logger.error(f'DB exist? error: {e}')
        return ret>0

    @classmethod
    def get_cafe(cls, db: DatabaseConfig, business_uid: str) -> Cafe:
        cafe: Cafe = None
        try:
            cafe = db.cafes.find_one({"business_uid": business_uid})
        except Exception as e:
            logger.error(f'DB exist? error: {e}')
        return cafe

    @classmethod
    def save_menu(cls, db: DatabaseConfig, business_uid:str, new_menu: Menu) -> bool:
        '''
            Save menu to new or existing cafe. 
            
            @return:
                ok: bool
                msg: error message if any
        '''
        cafe_data: Cafe = cls.get_cafe(db,business_uid=business_uid)
        ok: bool = False
        msg: str = ""
        try:
            if cafe_data:
                # Update the existing Cafe document
                cafe = Cafe.from_dict(cafe_data)
                cafe.menus.append(new_menu)
                db.cafes.update_one({"business_uid": business_uid}, {"$set": cafe.to_dict()})
            else:
                # Create a new Cafe object and insert it into MongoDB
                new_cafe = Cafe(business_uid=business_uid, menus=[new_menu])
                db.cafes.insert_one(new_cafe.to_dict())
            ok = True
        except Exception as e:
            logger.error(f'Error saving new manu: {e}')
            msg = str(e)
        
        return ok, msg
    

