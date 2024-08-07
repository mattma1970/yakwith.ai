from attrs import define, field
from menus import BaseMenuHandler


@define
class MenuManager:
    """
    Class to handle all Menu operations
    """

    cafe_id: str = field()
    menu_handler: BaseMenuHandler = field()
