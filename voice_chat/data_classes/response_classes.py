"""
Convenience classes for return types for fastAPi endpoints.
"""

from typing import Dict, Any
from dataclasses import dataclass
import base64
import json


@dataclass
class StdResponse:
    ok: bool = False
    msg: str = ""
    payload: Any = None

    def to_dict(
        self,
    ) -> Dict:
        return {
            "status": "success" if self.ok else "error",
            "msg": self.msg,
            "payload": self.payload,
        }

    def to_string(self) -> str:
        res = {
            "status": "success" if self.ok else "error",
            "msg": self.msg,
            "payload": self.payload,
        }
        return json.dumps(res)


@dataclass
class MultiPartResponse:
    """Use for sending metadata and audio"""

    json_data: str
    audio_bytes: bytes
    boundary: str = "frame"  # BOundery marker for multipart mixed type response for fastAPI endpoints

    def prepare(self) -> str:
        """
        return string with boundary markers and data.
        """
        return f"--{self.boundary}\r\nContent-Type: application/json\r\n\r\n{self.json_data}\r\n{self.boundary}--frame\r\nContent-Type: audio/mpeg\r\n\r\n {base64.b64encode(self.audio_bytes).decode('utf-8')}"


@dataclass
class BlendShapesMultiPartResponse:
    """Use for sending metadata and audio"""

    request_uid: str
    blendshapes: str
    json_data: str
    audio_bytes: bytes
    boundary: str = "frame"  # BOundery marker for multipart mixed type response for fastAPI endpoints

    def prepare(self) -> str:
        """
        return string with boundary markers and data.
        """
        return f"""--{self.boundary}\r\nContent-Type: application/json\r\nType: blendshapes\r\nID: {self.request_uid}\r\n\r\n{self.blendshapes}\r\n{self.boundary}
                   --{self.boundary}\r\nContent-Type: application/json\r\nType: visemes\r\nID: {self.request_uid}\r\n\r\n{self.json_data}\r\n{self.boundary}
                    --{self.boundary}\r\nContent-Type: audio/mpeg\r\nID: {self.request_uid}\r\n\r\n {base64.b64encode(self.audio_bytes).decode('utf-8')}
                """
