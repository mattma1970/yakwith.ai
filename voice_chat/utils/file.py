import os, errno
from typing import Union, Dict, List
from datetime import datetime


@staticmethod
def createIfMissing(file_name: Union[str, os.PathLike]) -> None:
    if os.path.exists(path=file_name) and os.path.isfile(file_name):
        return True
    else:
        os.makedirs(os.path.dirname(file_name))
        with open(file_name, "w") as f:
            f.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
