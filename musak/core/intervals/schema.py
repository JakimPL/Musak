from pydantic import BaseModel

from musak.config.defaults import HIGHEST_NOTE, LOWEST_NOTE, SEQUENTIAL, TEMPO
from musak.core.notation.schema import ScoreData
from musak.core.schemas.common import ConfigResponse


class IntervalRequest(BaseModel):
    tempo: int = TEMPO
    lowest_note: int = LOWEST_NOTE
    highest_note: int = HIGHEST_NOTE
    sequential: bool = SEQUENTIAL
    intervals: dict[str, int] = {}  # empty means use all enabled from config


class IntervalResponse(BaseModel):
    directory: str
    audio_source: str
    score_data: ScoreData
    interval_info: str
    intervals: dict[str, int]


# Alias for the config endpoint response
IntervalConfigResponse = ConfigResponse
