""" Misc. useful utilities"""

""" 
This software is licensed under the Creative Commons Attribution 4.0 International License.

Copyright 2024 MatthewMa mtman@yakwith.ai

Licensed under the Creative Commons Attribution 4.0 International License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://creativecommons.org/licenses/by/4.0/

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import random
import string

from uuid import uuid4


def random_string(length=6):
    """Generate a random string of ASCII characters."""
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


def get_uid(length=-1) -> str:
    uid = str(uuid4())
    if length > 0 and length < len(uid):
        uid = uid[:length]
    return uid
