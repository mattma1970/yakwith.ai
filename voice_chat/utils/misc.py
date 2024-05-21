import random
import string
from datetime import datetime

from uuid import uuid4


def random_string(length=6):
    """Generate a random string of ASCII characters."""
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


def order_prefix(length=3):
    return "".join(str(datetime.timestamp()), random_string(3))


def get_uid(length=-1) -> str:
    uid = str(uuid4())
    if length > 0 and length < len(uid):
        uid = uid[:length]
    return uid
