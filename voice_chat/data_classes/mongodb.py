from typing import List, Dict, Union, Tuple, Any
from pymongo import MongoClient
import logging
import os

from omegaconf import OmegaConf
import base64
from itertools import chain
import json

from voice_chat.data_classes.data_models import Menu, Cafe, ImageSelector, ModelChoice

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
        self.data = self.db[config.database.data_collection]


class DataHelper:
    """
    Miscellaneous operations n the data collection
    """

    def __init__(self):
        pass

    @classmethod
    def get_non_business_data(
        cls,
        config: DatabaseConfig,
        *,
        table_name: str,
        return_field_names: List[str],
        # sort: bool = False,
        # sort_field_name: str  # can't be a dictionary field name.
    ):
        """Get a list of str fields for given table."""
        selector: Dict = {}
        if config.data.find({"table_name": {"$exists": True}}):
            for field_name in return_field_names:
                config.data.find({f"{field_name}": {"$exists": True}})
                selector[f"{field_name}"] = True
            selector["_id"] = False  # drop it because its not seriaizable
        else:
            return None
        results = config.data.find({"table_name": table_name}, selector)
        return results


class ServicesHelper:
    """
    Miscellaneous CRUD functions used by the Services endpoints.
    For non-business specifc models e.g. for text editing
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
                    logger.error(f"Failed to insert images into menu object: {e}")

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
    def upsert_cafe_settings(
        cls,
        db: DatabaseConfig,
        business_uid: str,
        updated_partial_cafe: Cafe,
        skip_fields: List[str] = ["menus"],
    ):
        """Update all fields except menus. Menus are not updated via settings UI."""
        ok: str = False
        msg: str = ""
        try:
            cafe: Cafe = cls.get_cafe(db, business_uid=business_uid)
            if cafe is None:
                cafe = updated_partial_cafe
            else:
                for key, value in updated_partial_cafe.__dict__.items():
                    if skip_fields is None or key not in skip_fields:
                        setattr(cafe, key, value)
            db.cafes.update_one(
                {"business_uid": business_uid}, {"$set": cafe.to_dict()}, upsert=True
            )
            ok = True
        except Exception as e:
            msg = f"A problem occured when upserting cafe settings for business_uid {business_uid}: {e}"
            logger.error(msg)
        return ok, msg

    @classmethod
    def parse_dict(
        cls,
        target: Union[Dict, str],
        keys: Union[str, List[str]] = None,
        drop_keys: bool = False,
    ) -> Union[Dict, str]:
        """
        Parse and Filter a dictionary or json string.

        args:
            target: Union[dict|str] The dictionary which is either a dict object or a stringified version.
            keys: Union[str],list]: Dictionary key or list of keys.
                                    If 'None' then return return the entire dictionary.
        @args:
            dictionary
        """
        ret: Any = None
        _target: Dict = None

        # Parse the target dictionary
        if isinstance(target, str):
            try:
                _target = json.loads(target)
            except Exception as e:
                logger.error(f"Error parsing dictionary (dict_parse). {e}")
        elif isinstance(target, Dict):
            _target = target

        # Deal with multiplicity of keys.
        if keys is None:
            ret = _target
        else:
            if not (isinstance(keys, list)):
                _keys = [keys]
            if drop_keys:
                ret = _target
            else:
                ret = {}
            for key in keys:
                if key in _target:
                    if drop_keys and key in ret:
                        del ret[key]
                    else:
                        ret[key] = target[key]
        return ret

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
    def get_menu_list(
        cls, db: DatabaseConfig, business_uid: str, for_display: bool = True
    ) -> List[Menu]:
        """
        Get the list of menus for a given business.

        @args:
            for_display: bool: if True then only the primary menus in a collection will be shown.
                               i.e. those with menu.collection.seqeunce_number>0 will be filtered out.
        @returns:
            filtered list of Menu objects.
        """
        ret: List[Menu] = []
        try:
            cafe: Cafe = cls.get_cafe(db, business_uid)
            ret = cafe.menus
            if for_display:

                def robust_filter(x: Menu):
                    if "sequence_number" in x.collection:
                        if int(x.collection["sequence_number"]) == 0:
                            return True
                        else:
                            return False
                    else:
                        return (
                            True  # include menus that are not members of a collection
                        )

                ret = list(filter(lambda x: robust_filter(x), ret))
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
                msg = "Erorr: Failed to delete menu."
        except Exception as e:
            logger.error(f"Failed to delete menu with err: {str(e)}")
            ok = False
            msg = f"Erorr: Failed to delete menu {str(e)}"

        return ok, msg

    @classmethod
    def get_menu_image_names(cls, menu: Menu) -> List[str]:
        """Get just the images file names"""
        return [
            menu.raw_image_rel_path,
            menu.thumbnail_image_rel_path,
            menu.ocr_image_rel_path,
        ]

    @classmethod
    def delete_all_images_by_menu_id(
        cls,
        db: DatabaseConfig,
        business_uid: str,
        menu_id: str,
        image_root: os.PathLike,
    ) -> Tuple[bool, str]:
        """Delete all images in the same group (specified by grp_id) as the menu page with menu_id == menu_id"""
        ok = False
        msg = ""
        try:
            cafe_dict: Dict = db.cafes.find_one(
                {"business_uid": business_uid, "menus.menu_id": menu_id}
            )
            cafe: Cafe = Cafe.from_dict(cafe_dict)
            menus = [menu for menu in cafe.menus if menu.menu_id == menu_id]
            images_to_delete: List[
                str
            ] = []  # list of file names of images to be deleted.
            if len(menus) > 0:
                grp_id: str = ""
                if "grp_id" in menus[0].collection:
                    grp_id = menus[0].collection["grp_id"]
                    images_to_delete = [
                        cls.get_menu_image_names(menu)
                        for menu in cafe.menus
                        if "grp_id" in menu.collection
                        and menu.collection["grp_id"] == grp_id
                    ]  # For backwards compat, some early menu's did not have grp id.
                    images_to_delete = sum(images_to_delete, [])
                else:
                    images_to_delete = cls.get_menu_image_names(menus[0])
                images_to_delete = [
                    img_rel_path
                    for img_rel_path in images_to_delete
                    if img_rel_path != ""
                ]
                logger.info(
                    f"Deleting {len(images_to_delete)} images in collection of images containing menu_id {menu_id}"
                )
                # Delete the image files.
                for image_file in images_to_delete:
                    if os.path.exists(os.path.join(image_root, image_file)):
                        os.remove(os.path.join(image_root, image_file))
            ok = True
        except Exception as e:
            logger.error(
                f"Failed to remove all images associated with menu grp_id {grp_id}: {e}"
            )
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
        """
        Update a single menu. Menu contains optional fields and so if these are
        not present, then the current value of the menu to be kept.
        """
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

    @classmethod
    def count_menus_in_collection(
        cls, db: DatabaseConfig, business_uid: str, grp_id: str
    ) -> int:
        if (
            grp_id and len(grp_id) <= 10
        ):  # TODO - hack to detect whether a UUID4 str wasn't passed in.
            return 0
        else:
            cafe: Cafe = cls.get_cafe(db, business_uid=business_uid)
            if len(cafe.menus) > 0:
                menus: List[Menu] = [
                    menu
                    for menu in cafe.menus
                    if "grp_id" in menu.collection
                    and menu.collection["grp_id"] == grp_id
                ]
                len(menus)
            return len(menus)

    @classmethod
    def collate_text(
        cls, db: DatabaseConfig, business_uid: str, grp_id: str
    ) -> Tuple[bool, str, int, str]:
        """
        Collect the text from all Menu records in the same grp and collate them in the Menu
        with collection.sequence_number == 0

        @args:
            db: database
            business_uid: str : global identifier of the current business being used.
            grp_id: str: group identifier for the collaction of images belonging to the same menu. menu.collection.grp_id
        @returns:
            ok: bool : success status of operation
            msg: str: error message, if any
            count: int: the number of images in the collection
            primary_menu_id: the menu_id of the menu in the collection that has sequence_numer == 0
        """

        menus: List[Menu] = cls.get_menu_list(db, business_uid, for_display=False)
        if len(menus) == 0:
            return (
                False,
                f"No business found with business_uid == {business_uid}",
                0,
                "",
            )

        def robust_filter(x: Menu) -> bool:
            if "grp_id" in x.collection:
                return x.collection["grp_id"] == grp_id
            else:
                return False

        primary_menu_id: str = ""
        menus = list(
            filter(lambda x: robust_filter(x), menus)
        )  # Filter out menus that might not have a group_id (legacy)
        sorted_menus = sorted(
            menus, key=lambda x: int(x.collection["sequence_number"])
        )  # Ensure they are in ascending seqeunce_number order ( the order they were added)
        if len(sorted_menus) > 0:
            primary_menu_id = sorted_menus[0].menu_id

        all_text = "".join([menu.menu_text for menu in sorted_menus])
        count = len(menus)
        ok: bool = False
        msg: str = ""

        ok, msg = cls.update_menu_field(
            db,
            business_uid=business_uid,
            menu_id=primary_menu_id,
            value=all_text,
            field="menu_text",
        )
        if not (ok):
            count = 0
            primary_menu_id = ""
            logger.error(f"collate_text failed for grp_id=={grp_id} ")

        return True, msg, count, primary_menu_id


class ModelHelper:
    """Helper functions for models collection which stored details of the model used by the establishment."""

    @classmethod
    def get_model_by_id(cls, db: DatabaseConfig, id: str) -> ModelChoice:
        model: ModelChoice = None
        try:
            model_details: Dict = db.data.find_one({"id": id, "table_name": "models"})
            model = ModelChoice.from_dict(model_details)
        except Exception as e:
            logger.error(f"No record found for model id: {id}, error {e}")
        return model

    def upsert_from_dict(cls, db: DatabaseConfig, model_details: Dict):
        msg: str = ""
        ok: bool = False
        try:
            db.data.update_one(
                {"id": model_details["id"], "table_name": "models"},
                {"$set": ModelChoice.from_dict(model_details)},
                upsert=True,
            )
        except Exception as e:
            msg = f"Failed to add model to model collection: {e}"
            logger.error(msg)
        return ok, msg
