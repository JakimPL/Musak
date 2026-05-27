from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.tokens.schema import MAX_ACCIDENTAL, MIN_ACCIDENTAL, MIN_DEGREE


class ChordQuality(StrEnum):
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"


class ChordExtension(StrEnum):
    TRIAD = "triad"
    SEVENTH = "seventh"
    NINTH = "ninth"
    ELEVENTH = "eleventh"
    FLAT_NINTH = "flat_ninth"
    SHARP_ELEVENTH = "sharp_eleventh"


DEFAULT_CHORD_EXTENSION: Final[ChordExtension] = ChordExtension.TRIAD


class Chord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    root_degree: int = Field(ge=MIN_DEGREE)
    root_accidental: int = Field(ge=MIN_ACCIDENTAL, le=MAX_ACCIDENTAL)
    quality: ChordQuality
    extension: ChordExtension = DEFAULT_CHORD_EXTENSION
