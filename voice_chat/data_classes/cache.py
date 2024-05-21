"""Abstract Class Definition for Cache"""

from abc import ABC, abstractclassmethod
from enum import Enum


class QueryType(Enum):
    REQUEST = 0
    RESPONSE = 1


class CacheHelper(ABC):
    def __init__(self):
        self._db = None
        self._cache_client = None

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, val):
        self._db = val

    @property
    def cache_client(self):
        return self._cache_client

    @cache_client.setter
    def cache_client(self, value):
        self._cache_client = value

    @abstractclassmethod
    def get_cache_key(self, query_type: QueryType, voice: str, text: str, id: str = ""):
        pass

    @abstractclassmethod
    def append_to_cache(self, key, subkey, value):
        pass
