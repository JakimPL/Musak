from typing import Optional

from pydantic import BaseModel

from core.schemas.common import ConfigResponse

NoteValue = int | tuple[int, int]


class RhythmRequest(BaseModel):
    tempo: int = 120
    groups: int = 1
    measures: int = 2
    time_signature: tuple[int, int] = (4, 4)
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
