import os
from typing import Union
from datetime import datetime


@staticmethod
def createIfMissing(file_name: Union[str, os.PathLike]) -> bool:
    if os.path.exists(path=file_name) and os.path.isfile(file_name):
        return True
    else:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            f.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            f.write("\n==========================\n\r\n\r")
        return False


@staticmethod
def createFolderIfMissing(folder_name: Union[str, os.PathLike]) -> bool:
    folder_name = folder_name.rstrip("/") + "/"
    if os.path.exists(path=folder_name):
        return True
    else:
        os.makedirs(os.path.dirname(folder_name), exist_ok=True)
        return False
