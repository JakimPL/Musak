from enum import StrEnum
from fractions import Fraction
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict


class DurationClass(StrEnum):
    WHOLE = "whole"
    HALF = "half"
    DOTTED_HALF = "dotted_half"
    QUARTER = "quarter"
    DOTTED_QUARTER = "dotted_quarter"
    EIGHTH = "eighth"
    DOTTED_EIGHTH = "dotted_eighth"
    SIXTEENTH = "sixteenth"
    TRIPLET_EIGHTH = "triplet_eighth"


DURATION_FRACTIONS: Final[dict[DurationClass, Fraction]] = {
    DurationClass.WHOLE: Fraction(1, 1),
    DurationClass.HALF: Fraction(1, 2),
    DurationClass.DOTTED_HALF: Fraction(3, 4),
    DurationClass.QUARTER: Fraction(1, 4),
    DurationClass.DOTTED_QUARTER: Fraction(3, 8),
    DurationClass.EIGHTH: Fraction(1, 8),
    DurationClass.DOTTED_EIGHTH: Fraction(3, 16),
    DurationClass.SIXTEENTH: Fraction(1, 16),
    DurationClass.TRIPLET_EIGHTH: Fraction(1, 12),
}

HALVED_DURATIONS: Final[dict[DurationClass, DurationClass]] = {
    DurationClass.WHOLE: DurationClass.HALF,
    DurationClass.HALF: DurationClass.QUARTER,
    DurationClass.DOTTED_HALF: DurationClass.DOTTED_QUARTER,
    DurationClass.QUARTER: DurationClass.EIGHTH,
    DurationClass.DOTTED_QUARTER: DurationClass.DOTTED_EIGHTH,
    DurationClass.EIGHTH: DurationClass.SIXTEENTH,
}

DOUBLED_DURATIONS: Final[dict[DurationClass, DurationClass]] = {
    shorter: longer for longer, shorter in HALVED_DURATIONS.items()
}


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


class NoteToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["note"] = "note"
    degree: int
    accidental: int
    octave_offset: int
    duration: DurationClass


class RestToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["rest"] = "rest"
    duration: DurationClass


class BarToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["bar"] = "bar"


class EndToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["end"] = "end"


Token = NoteToken | RestToken | BarToken | EndToken
