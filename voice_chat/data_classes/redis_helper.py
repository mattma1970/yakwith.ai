"""
Helper and convenience functions for Redis cache.

"""

import redis
import logging
import os
import re
import pickle

logger = logging.getLogger(__name__)


class RedisHelper:
    def __init__(self, db=0):
        self.db_name = db
        self.redis_client = redis.Redis(
            host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"], db=db
        )
        self.running = False  # Flag indicating there is a running redis server.
        try:
            self.redis_client.ping()
            self.running = True
        except RuntimeError:
            logger.error(
                "Redis server is not running or could not be connected to.Please check."
            )

    def get_cache_key(self, key_type: str, voice: str, text: str, id: str = ""):
        """
        Build cache keys. Standardize formating of keys.
        @args:
            key_type: str: ['res','req'] where res indicates is a responses from LLM, and req indicates is a prompt (request)
            voice: voice ID for TTS
            text: text string being cached
            id: str: use case specific ID. e.g menu_id or leave blank to enable it to be used for every use case.
        """
        return (
            f"{key_type.upper()}:{voice.upper()}:{id.upper()}:::{self.safe_key(text)}"
        )

    def select_database(redis_client, db):
        """If the database is persisted and that database has db value other than 0, then you'll need to select the database manaully."""
        redis_client.select(db)

    def safe_key(self, key):
        return re.sub(r"[^a-zA-Z0-9,\s\.]", "", key)

    # CRUD string datatypes
    def create(self, key, value):
        return self.redis_client.set(key, value)

    def read(self, key):
        return self.redis_client.get(key)

    def update(self, key, value):
        if self.redis_client.exists(key):
            return self.redis_client.set(key, value)
        else:
            return False

    def delete(self, key):
        return self.redis_client.delete(key)

    def exists(self, key):
        return self.redis_client.exists(key)

    # Hash datatypes ( python dictionaries)
    def hset(self, name, key, value):
        return self.redis_client.hset(name, key, value)

    # Function to create or append to list types in a cache entry
    def append_to_cache(self, key, subkey, value):
        # Check if the key already exists in the cache
        if self.exists(key):
            # If the key exists, append to the dictionary value
            cached_value = self.hget(key, subkey)
            if cached_value:
                # If the subkey exists, append the value to the existing list
                cached_list = pickle.loads(cached_value)  # Deserialize the list
                cached_list.append(value)
                self.hset(key, subkey, pickle.dumps(cached_list))  # Update the cache
            else:
                # If the subkey does not exist, create a new list with the value

                self.hset(key, subkey, pickle.dumps([value]))
        else:
            # If the key does not exist, create a new dictionary with the subkey and value
            self.redis_client.hset(key, subkey, pickle.dumps([value]))

    def hget(self, name, key):
        return self.redis_client.hget(name, key)

    def hgetall(self, name):
        return self.redis_client.hgetall(name)

    def hdel(self, name, *keys):
        return self.redis_client.hdel(name, *keys)

    def hexists(self, name, key):
        return self.redis_client.hexists(name, key)

    def hlen(self, name):
        return self.redis_client.hlen(name)

    def hkeys(self, name):
        return self.redis_client.hkeys(name)
