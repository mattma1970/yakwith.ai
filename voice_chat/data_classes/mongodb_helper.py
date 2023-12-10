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

from omegaconf import OmegaConf
import base64
from itertools import chain

from voice_chat.data_classes.data_models import Menu, Cafe, ImageSelector

logger = logging.getLogger(__name__)

"""
    All classes related to MongoDB and pymongo.
"""


class DatabaseConfig:
    """Database connection for MongoDB"""

    def __init__(self, config: Dict):
        self.client = MongoClient(
            f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@{config.database.url}"
        )
        self.db = self.client[config.database.name]
        self.cafes = self.db[config.database.default_collection]
        self.services = self.db[config.database.services_collection]


class ServicesHelper:
    """
    Miscellaneous CRUD functions used by the Services endpoints.
    """

    def __init__(self):
        pass

    @classmethod
    def get_field_by_business_id(
        cls,
        config: DatabaseConfig,
        *,
        business_uid: str = None,
        field: str = "",
        flatten: bool = True,
    ):
        """
        Get a list column from the passed in config (database).

        @args:
            config: DatabaseConfig : mongodb config
            buisness_uid: business_uid for record
            field: str: field to return
            flatten: bool = True: if result is list of lists then flatten to simple list.
        """
        if not (config.services.find({f"{field}": {"$exists": True}})):
            return None

        if business_uid and isinstance(business_uid, str):
            results = config.services.find({"business_uid": business_uid})
            if results is not None and isinstance(results, Dict):
                return results[field]
            else:
                ret: Any = [result[field] for result in results]
                if flatten:
                    ret = list(chain.from_iterable(ret))
                return ret


class MenuHelper:
    def __init__(self):
        pass

    @classmethod
    def insert_images(
        cls,
        config: OmegaConf,
        menus: Union[Menu, List[Menu]],
        image_types: List[ImageSelector],
    ) -> Union[Menu, List[Menu]]:
        """Images are stored on disk outside the database. This functions add the images as base64, uts-8 encoded strings to the menu."""
        ret = None
        if isinstance(menus, Menu):
            _menus = [menus]
        else:
            _menus = menus
        for menu in _menus:
            for image_type in image_types:
                try:
                    if image_type == ImageSelector.RAW:
                        if menu.raw_image_rel_path != "":
                            with open(
                                f"{config.assets.image_folder}/{menu.raw_image_rel_path}",
                                "rb",
                            ) as image_file:
                                menu.raw_image_data = base64.b64encode(
                                    image_file.read()
                                ).decode("utf-8")
                    elif image_type == ImageSelector.OCR:
                        if menu.ocr_image_rel_path:
                            with open(
                                f"{config.assets.image_folder}/{menu.ocr_image_rel_path}",
                                "rb",
                            ) as image_file:
                                menu.raw_image_data = base64.b64encode(
                                    image_file.read()
                                ).decode("utf-8")
                    elif image_type == ImageSelector.THUMBNAIL:
                        if menu.thumbnail_image_rel_path:
                            with open(
                                f"{config.assets.image_folder}/{menu.thumbnail_image_rel_path}",
                                "rb",
                            ) as image_file:
                                menu.thumbnail_image_data = base64.b64encode(
                                    image_file.read()
                                ).decode("utf-8")
                except Exception as e:
                    logger.error("Failed to insert images into menu object: {e}")

                if isinstance(menus, Menu):
                    """If only a single Menu was passed in then return a Menu object"""
                    ret = _menus[0]
                else:
                    ret = _menus

        return ret

    @classmethod
    def cafe_exists(cls, db: DatabaseConfig, business_uid: str) -> bool:
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
            logger.error(f"No match business found: {e}")
        return cafe

    @classmethod
    def get_one_menu(cls, db: DatabaseConfig, business_uid: str, menu_id: str) -> Menu:
        ret: Menu = None
        try:
            cafe: Cafe = cls.get_cafe(db, business_uid, {"menus.menu_id": menu_id})
            ret = [menu for menu in cafe.menus if menu.menu_id == menu_id][0]
        except Exception as e:
            logger.warning(f"DB record for cafe {business_uid} not found: {e}")
        return ret

    @classmethod
    def get_menu_list(cls, db: DatabaseConfig, business_uid: str) -> List[Menu]:
        ret: List[Menu] = []
        try:
            cafe: Cafe = cls.get_cafe(db, business_uid)
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
                new_cafe: Cafe = Cafe(
                    business_uid=business_uid,
                    menus=[Menu.from_dict(new_menu.to_dict())],
                )  # Hack to overcome corruption of new_menu> Maybe due to how pydantic deals with nested dataclasses?
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
        """Set the menu.f'{field}'= value"""
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

    @classmethod
    def update_menu(
        cls, db: DatabaseConfig, business_uid: str, updated_menu: Menu
    ) -> Tuple[bool, str]:
        """Update a single menu. Menu contains optional fields and so if these are not present, then the value if the menu to be updated is applied"""
        ok: bool = False
        msg: str = ""
        try:
            cafe: Cafe = cls.get_cafe(
                db, business_uid, {"menus.menu_id": updated_menu.menu_id}
            )  # Double check
            _menus = []
            for menu in cafe.menus:
                if menu.menu_id == updated_menu.menu_id:
                    tmp = updated_menu.to_dict()
                    tmp = menu.to_dict() | tmp
                    _menus.append(tmp)
                else:
                    _menus.append(menu.to_dict())
            db.cafes.update_one(
                {"business_uid": business_uid}, {"$set": {"menus": _menus}}
            )
            ok = True
        except Exception as e:
            msg = f"Error updating one menu: {e}"
            logger.error(msg)
        return ok, msg
