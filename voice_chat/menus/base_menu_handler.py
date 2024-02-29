from attrs import define, field, Factory
from typing import Optional, List, Dict, Union
from data_classes.chat_data_classes import Menu, MenuList
from datetime import datetime
from abc import ABC, abstractmethod
from voice_chat.utils import time_only


class BaseMenuHandler(ABC):
    cafe_id: str = field()

    @classmethod
    def menu_is_active(cls, menu: Menu, query_time: datetime) -> bool:
        """Check the menu is active at the query time."""
        date_valid = (
            datetime.strptime(menu.start_date, "%Y-%m-%d") <= query_time
        ) and (datetime.strptime(menu.end_date, "%Y-%m-%d") >= query_time)
        time_valid = (
            time_only(datetime.strptime(menu.start_time_of_day, "%H:%M"))
            <= time_only(query_time)
        ) and (
            time_only(datetime.strptime(menu.end_time_of_day, "%H:%M"))
            >= time_only(query_time)
        )
        return date_valid and time_valid

    @abstractmethod
    def get_menu(self) -> Union[Menu, MenuList]: ...

    @abstractmethod
    def create(self) -> str:
        """Returns a uuid4 str"""
        ...

    @abstractmethod
    def update(self) -> bool: ...

    @abstractmethod
    def delete(self) -> bool: ...
