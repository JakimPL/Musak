from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from musak.config.defaults import (
    HIGHEST_NOTE,
    LOWEST_NOTE,
    MAX_HIGHEST_NOTE,
    MAX_LOWEST_NOTE,
    MAX_TEMPO,
    MIN_HIGHEST_NOTE,
    MIN_LOWEST_NOTE,
    MIN_TEMPO,
    SEQUENTIAL,
    TEMPO,
)
from musak.core.schemas.common import ConfigResponse
from musak_shared.notation.schema import ScoreData


class InversionRequest(BaseModel):
    tempo: int = Field(default=TEMPO, ge=MIN_TEMPO, le=MAX_TEMPO)
    lowest_note: int = Field(default=LOWEST_NOTE, ge=MIN_LOWEST_NOTE, le=MAX_LOWEST_NOTE)
    highest_note: int = Field(default=HIGHEST_NOTE, ge=MIN_HIGHEST_NOTE, le=MAX_HIGHEST_NOTE)
    sequential: bool = Field(default=SEQUENTIAL)
    chords: dict[str, list[int]] = Field(default_factory=dict)


class InversionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_data: str = ""
    score_data: ScoreData
    chord_info: dict[str, Any] = Field(default_factory=dict)
    chord_types: list[str]
    inversions_numbers: dict[str, int]


InversionConfigResponse = ConfigResponse
