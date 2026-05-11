from pydantic import BaseModel
from core.schemas.common import ConfigResponse


class InversionRequest(BaseModel):
    tempo: int = 120
    lowest_note: int = 40
    highest_note: int = 90
    sequential: bool = False
    chords: dict[str, list[int]] = {}  # empty means use all enabled from config


class InversionResponse(BaseModel):
    directory: str
    audio_source: str
    image_source: str
    chord_info: str
    chord_types: list[str]
    inversions_numbers: dict[str, int]


InversionConfigResponse = ConfigResponse
