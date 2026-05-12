from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak.modules.elements.time_signature import (
    TimeSignatureType,
    validate_time_signature,
)

Clef = Literal["treble", "bass", "alto", "tenor", "percussion"]
VexflowDuration = Literal[
    "w",
    "h",
    "q",
    "8",
    "16",
    "32",
    "wr",
    "hr",
    "qr",
    "8r",
    "16r",
    "32r",
]

WHOLE: Final[VexflowDuration] = "w"
HALF: Final[VexflowDuration] = "h"
QUARTER: Final[VexflowDuration] = "q"
EIGHTH: Final[VexflowDuration] = "8"
SIXTEENTH: Final[VexflowDuration] = "16"
THIRTY_SECOND: Final[VexflowDuration] = "32"
REST_SUFFIX: Final[str] = "r"


class NoteData(BaseModel):
    model_config = ConfigDict(frozen=True)

    keys: list[str]
    duration: VexflowDuration
    dots: Annotated[int, Field(ge=0, le=2)] = 0


class VoiceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    notes: list[NoteData]


class StaveData(BaseModel):
    model_config = ConfigDict(frozen=True)

    clef: Clef
    time_signature: TimeSignatureType | None = None
    voices: list[VoiceData]

    @field_validator("time_signature")
    @classmethod
    def check_time_signature(cls, value: TimeSignatureType | None) -> TimeSignatureType | None:
        if value is not None:
            validate_time_signature(value)

        return value


class ScoreData(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: list[list[StaveData]]
    tempo: Annotated[int, Field(gt=0)] | None = None
    max_notes_per_measure: int | None = None
