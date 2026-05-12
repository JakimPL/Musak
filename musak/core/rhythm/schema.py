from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from musak.config.defaults import (
    GROUPS,
    MAX_GROUPS,
    MAX_MEASURES,
    MAX_TEMPO,
    MEASURES,
    MELODIC,
    MIN_GROUPS,
    MIN_MEASURES,
    MIN_TEMPO,
    TEMPO,
    TIME_SIGNATURE,
)
from musak.core.notation.schema import ScoreData
from musak.core.schemas.common import ConfigResponse

NoteValue = int | tuple[int, int]


class RhythmRequest(BaseModel):
    tempo: int = Field(default=TEMPO, ge=MIN_TEMPO, le=MAX_TEMPO)
    groups: int = Field(default=GROUPS, ge=MIN_GROUPS, le=MAX_GROUPS)
    measures: int = Field(default=MEASURES, ge=MIN_MEASURES, le=MAX_MEASURES)
    time_signature: tuple[int, int] = Field(default=TIME_SIGNATURE)
    melodic: bool = Field(default=MELODIC)
    notes: list[NoteValue] = Field(default_factory=list)
    phrases: list[list[NoteValue]] = Field(default_factory=list)
    custom_phrases: str = Field(default="")


class RhythmResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    directory: str = ""
    audio_source: str = ""
    score_data: ScoreData | None = None
    exception: Optional[str] = None
    time_signature_error: bool = False


RhythmConfigResponse = ConfigResponse
