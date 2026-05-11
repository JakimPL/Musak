from pydantic import BaseModel
from core.schemas.common import ConfigResponse


class IntervalRequest(BaseModel):
    tempo: int = 120
    lowest_note: int = 40
    highest_note: int = 90
    sequential: bool = False
    intervals: dict[str, int] = {}  # empty means use all enabled from config


class IntervalResponse(BaseModel):
    directory: str
    audio_source: str
    image_source: str
    interval_info: str
    intervals: dict[str, int]


# Alias for the config endpoint response
IntervalConfigResponse = ConfigResponse
