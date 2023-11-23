"""
Utilities to read json data stored in data_proxies. 
This is temporary solution and the data will eventually be moved to a database
"""

from typing import List, Dict, Optional, Any
import os
from os import PathLike
import json


class DataProxy:
    DATA_PROXIES_ROOT_FOLDER = (
        "/home/mtman/Documents/Repos/yakwith.ai/voice_chat/data_proxies"
    )

    @classmethod
    def get_all(
        cls, data_source_name: str, root: str = DATA_PROXIES_ROOT_FOLDER
    ) -> List[Dict]:
        """select * from data_proxies.data_filename"""
        data = None
        fqn = os.path.join(root, f"{data_source_name}.json")
        print(f"{fqn}")
        if os.path.exists(fqn):
            try:
                with open(fqn, "r") as fp:
                    data = json.load(fp)
            except Exception as e:
                raise FileNotFoundError(f"Problem loading {e}")
        else:
            raise RuntimeError(f"Cant find data_proxy : {fqn}")
        return data

    @classmethod
    def verify_client_authorized(
        cls,
        authorization_key: str,
        data_source_name: str = "authorized_clients",
        root: str = DATA_PROXIES_ROOT_FOLDER,
    ) -> bool:
        """Check the passed in key is valid"""
        clients = cls.get_all(data_source_name, root)
        for client in clients["authorized_clients"]:
            if authorization_key == client["authorization_key"]:
                return True
        return False

    @classmethod
    def get_3p_service_configs(
        cls,
        authorization_key: str,
        authorization_data_source_name: str,
        service_name: str,
        service_data_source_name: str,
        root: str = DATA_PROXIES_ROOT_FOLDER,
    ) -> Dict:
        """Get the settings for the 3rd party api"""
        response = None

        authorized = cls.verify_client_authorized(
            authorization_key, authorization_data_source_name, root=root
        )
        if not authorized:
            raise RuntimeError("Rquest for service key from unauthorized client.")

        configs = cls.get_all(service_data_source_name, root)
        for config in configs["service_configs"]:
            if config["name"] == service_name:
                response = config
        return response


if __name__ == "__main__":
    print(DataProxy.get_all("service_conafigs"))
    print(len(DataProxy.get_all("service_configs")))
