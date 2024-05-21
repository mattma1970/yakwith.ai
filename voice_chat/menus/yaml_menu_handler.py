from attrs import define
import os
import glob
from typing import Optional
from data_classes.chat_data_classes import Menu, MenuList
from omegaconf import OmegaConf
from datetime import datetime
from voice_chat.menus import BaseMenuHandler


@define
class YAMLMenuHandler(BaseMenuHandler):
    """
    Handle menus stored as YAML files on local disk. OmegaConf used to deal with YAML.

    Arguments:
        path: (str, *optional*): use this to override the default location where menus stored on disk
        self.menu_filename (str, *optional*): name of the menu file.
        name (str, *optional*): use to select menu.name menu with
        use_time_of_day (bool, *optional*): if True then the menu should be selected from the MenuList using a time validity fields in the Menu. Ignored if name is set

    Usage:
        Single Menu file: If there is only a single yaml file at path then it is used as the menu source file. Specifying a self.menu_filename will override this behaviour
        Selecting menu form MenuList: If the self.menu_filename is specified and the file contains a list of Menus then a sinlge menu is selected according to the
            name (if set). If the name is not set then the menu will be selected according to the first one in the list active according to the menu_start date
            and time fields in the menu and the current time.
    """

    path: os.PathLike = (None,)
    menu_filename: Optional[str] = (None,)
    name: Optional[str] = (None,)
    use_time_of_day: Optional[bool] = (True,)

    def get_menu(self) -> Menu:
        """Get menu for cafe. Menu might be single menu file or a list of menus"""
        selected_menu: Menu = None

        if self.path is None:
            raise RuntimeError("a path must specified when sourcing menus from disk.")

        if self.menu_filename is None:
            pattern = os.path.join(self.path, self.cafe_id, "*.yaml")
            filelist = glob.glob(pattern)
            if len(filelist) > 1:
                raise RuntimeError(
                    f"menu folder for {self.cafe_id} contains more than 1 yaml file. Please pass in self.menu_filename"
                )
            self.menu_filename = filelist[
                0
            ]  # If there's only one yaml file then use it.

        if "." not in self.menu_filename:
            self.menu_filename += ".yaml"

        if self.menu_filename in filelist:
            raw_menus = OmegaConf.load(
                os.path.join(self.path, self.cafe_id, self.menu_filename)
            )
            if hasattr(raw_menus, "menus"):
                # Then its a MenuList
                raw_menus = MenuList.from_omega_conf(raw_menus)
                # Use either 'name' or 'time_of_day' to filter the list of menus.
                if self.name is not None:
                    for menu in raw_menus:
                        if menu.name == self.name:
                            selected_menu = menu
                    if selected_menu is None:
                        raise RuntimeError(
                            f"Menu names {self.name} not found in menu list saved as {self.menu_filename}"
                        )
                elif self.use_time_of_day:
                    for menu in raw_menus:
                        if self.menu_is_active(menu, datetime.now()):
                            selected_menu = menu
                            break
                else:
                    raise RuntimeError(
                        "One of menu name or use_time_of_day must to set in order to select a menu"
                    )
            else:  # its a single menu file
                selected_menu = Menu.from_omega_conf(
                    OmegaConf.load(
                        os.path.join(self.path, self.cafe_id, self.menu_filename)
                    )
                )
        else:
            raise RuntimeError(
                f"{self.menu_filename} not found in {os.path(self.path, self.cafe_id)}"
            )

        return selected_menu

    def create(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError
