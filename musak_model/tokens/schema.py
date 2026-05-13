from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field


class ScaleType(StrEnum):
    MAJOR = "major"
    NATURAL_MINOR = "natural_minor"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    LOCRIAN = "locrian"


SCALE_INTERVALS: Final[dict[ScaleType, tuple[int, ...]]] = {
    ScaleType.MAJOR: (0, 2, 4, 5, 7, 9, 11),
    ScaleType.NATURAL_MINOR: (0, 2, 3, 5, 7, 8, 10),
    ScaleType.HARMONIC_MINOR: (0, 2, 3, 5, 7, 8, 11),
    ScaleType.MELODIC_MINOR: (0, 2, 3, 5, 7, 9, 11),
    ScaleType.DORIAN: (0, 2, 3, 5, 7, 9, 10),
    ScaleType.PHRYGIAN: (0, 1, 3, 5, 7, 8, 10),
    ScaleType.LYDIAN: (0, 2, 4, 6, 7, 9, 11),
    ScaleType.MIXOLYDIAN: (0, 2, 4, 5, 7, 9, 10),
    ScaleType.LOCRIAN: (0, 1, 3, 5, 6, 8, 10),
}


class Hand(StrEnum):
    RIGHT = "right"
    LEFT = "left"


MIN_DEGREE: Final[int] = 1
MAX_DEGREE: Final[int] = 7

MIN_ACCIDENTAL: Final[int] = -1
MAX_ACCIDENTAL: Final[int] = 1

MIN_OCTAVE_OFFSET: Final[int] = -2
MAX_OCTAVE_OFFSET: Final[int] = 2
MIN_DURATION_ID: Final[int] = 0

RIGHT_HAND_HOME_OCTAVE: Final[int] = 4
LEFT_HAND_HOME_OCTAVE: Final[int] = 3

HAND_HOME_OCTAVES: Final[dict[Hand, int]] = {
    Hand.RIGHT: RIGHT_HAND_HOME_OCTAVE,
    Hand.LEFT: LEFT_HAND_HOME_OCTAVE,
}

MIN_DIFFICULTY_LEVEL: Final[int] = 1
MAX_DIFFICULTY_LEVEL: Final[int] = 6

VALID_BAR_COUNTS: Final[tuple[int, ...]] = (4, 8, 12, 16, 24, 32)

VALID_TIME_SIGNATURES: Final[tuple[tuple[int, int], ...]] = (
    (2, 4),
    (3, 4),
    (4, 4),
    (3, 8),
    (6, 8),
)


_ACCIDENTAL_SYMBOLS: Final[dict[int, str]] = {-1: "b", 0: "", 1: "#"}


class NoteToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["note"] = "note"
    degree: int
    accidental: int
    octave_offset: int
    duration_id: int = Field(ge=MIN_DURATION_ID)

    def __repr__(self) -> str:
        accidental = _ACCIDENTAL_SYMBOLS[self.accidental]
        register = f"[{self.octave_offset:+d}]" if self.octave_offset != 0 else ""
        return f"{self.degree}{accidental}{register}/d{self.duration_id}"

    def __str__(self) -> str:
        return repr(self)


class RestToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["rest"] = "rest"
    duration_id: int = Field(ge=MIN_DURATION_ID)

    def __repr__(self) -> str:
        return f"r/d{self.duration_id}"

    def __str__(self) -> str:
        return repr(self)


class BarToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["bar"] = "bar"

    def __repr__(self) -> str:
        return "|"

    def __str__(self) -> str:
        return repr(self)


class EndToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["end"] = "end"

    def __repr__(self) -> str:
        return "‖"

    def __str__(self) -> str:
        return repr(self)


Token = NoteToken | RestToken | BarToken | EndToken
