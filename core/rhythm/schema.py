from typing import Optional

from pydantic import BaseModel

from config.defaults import (
    GROUPS,
    MEASURES,
    TEMPO,
    TIME_SIGNATURE_DENOMINATOR,
    TIME_SIGNATURE_NUMERATOR,
)
from core.schemas.common import ConfigResponse

NoteValue = int | tuple[int, int]


class RhythmRequest(BaseModel):
    tempo: int = TEMPO
    groups: int = GROUPS
    measures: int = MEASURES
    time_signature: tuple[int, int] = (
        TIME_SIGNATURE_NUMERATOR,
        TIME_SIGNATURE_DENOMINATOR,
    )
    notes: list[NoteValue] = []
    phrases: list[list[NoteValue]] = []
    custom_phrases: str = ""


class RhythmResponse(BaseModel):
    directory: str = ""
    audio_source: str = ""
    image_source: str = ""
    score: str = ""
    exception: Optional[str] = None
    time_signature_error: bool = False


RhythmConfigResponse = ConfigResponse
