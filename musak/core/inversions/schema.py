from pydantic import BaseModel

from musak.config.defaults import HIGHEST_NOTE, LOWEST_NOTE, SEQUENTIAL, TEMPO
from musak.core.notation.schema import ScoreData
from musak.core.schemas.common import ConfigResponse


class InversionRequest(BaseModel):
    tempo: int = TEMPO
    lowest_note: int = LOWEST_NOTE
    highest_note: int = HIGHEST_NOTE
    sequential: bool = SEQUENTIAL
    chords: dict[str, list[int]] = {}  # empty means use all enabled from config


class InversionResponse(BaseModel):
    directory: str
    audio_source: str
    score_data: ScoreData
    chord_info: str
    chord_types: list[str]
    inversions_numbers: dict[str, int]


InversionConfigResponse = ConfigResponse
