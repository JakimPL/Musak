from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from musak_model.tokens.symbols import (
    BAR_SYMBOL,
    DURATION_CLOSE_SYMBOL,
    DURATION_OPEN_SYMBOL,
    DURATION_SEPARATOR_SYMBOL,
    END_SYMBOL,
    HOLD_SYMBOL,
    JOIN_WITH_PREVIOUS_SYMBOL,
    LEFT_HAND_SYMBOL,
    OCTAVE_DOWN_SYMBOL,
    OCTAVE_UP_SYMBOL,
    REST_SYMBOL,
    RIGHT_HAND_SYMBOL,
    START_SYMBOL,
    TEXT_FLAT_SYMBOL,
    TEXT_SHARP_SYMBOL,
)
from musak_shared.ratios import format_ratio

if TYPE_CHECKING:
    from musak_model.tokens.duration import DurationVocabulary


class ScaleType(StrEnum):
    MAJOR = "major"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"


SCALE_INTERVALS: Final[dict[ScaleType, tuple[int, ...]]] = {
    ScaleType.MAJOR: (0, 2, 4, 5, 7, 9, 11),
    ScaleType.HARMONIC_MINOR: (0, 2, 3, 5, 7, 8, 11),
    ScaleType.MELODIC_MINOR: (0, 2, 3, 5, 7, 9, 11),
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

RIGHT_HAND_HOME_OCTAVE: Final[int] = 5
LEFT_HAND_HOME_OCTAVE: Final[int] = 3

HAND_HOME_OCTAVES: Final[dict[Hand, int]] = {
    Hand.RIGHT: RIGHT_HAND_HOME_OCTAVE,
    Hand.LEFT: LEFT_HAND_HOME_OCTAVE,
}

MIN_DIFFICULTY_LEVEL: Final[int] = 0

_ACCIDENTAL_SYMBOLS: Final[dict[int, str]] = {-1: TEXT_FLAT_SYMBOL, 0: "", 1: TEXT_SHARP_SYMBOL}
_TEXT_ACCIDENTAL_SYMBOLS: Final[dict[int, str]] = {-1: TEXT_FLAT_SYMBOL, 0: "", 1: TEXT_SHARP_SYMBOL}


def _duration_text(duration_id: int, *, duration_vocabulary: DurationVocabulary) -> str:
    duration = duration_vocabulary.id_to_fraction(duration_id)
    duration_text = format_ratio(duration, separator=DURATION_SEPARATOR_SYMBOL)
    return f"{DURATION_OPEN_SYMBOL}{duration_text}{DURATION_CLOSE_SYMBOL}"


def _octave_offset_text(octave_offset: int) -> str:
    if octave_offset > 0:
        return f"{OCTAVE_UP_SYMBOL}{octave_offset}"

    if octave_offset < 0:
        return f"{OCTAVE_DOWN_SYMBOL}{abs(octave_offset)}"

    return ""


class NoteToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["note"] = "note"
    degree: int
    accidental: int
    octave_offset: int
    duration_id: int = Field(ge=MIN_DURATION_ID)

    def to_text(self, *, duration_vocabulary: DurationVocabulary) -> str:
        accidental = _TEXT_ACCIDENTAL_SYMBOLS[self.accidental]
        octave_offset = _octave_offset_text(self.octave_offset)
        duration = _duration_text(self.duration_id, duration_vocabulary=duration_vocabulary)
        return f"{self.degree}{accidental}{octave_offset}{duration}"

    def __repr__(self) -> str:
        accidental = _ACCIDENTAL_SYMBOLS[self.accidental]
        register = f"[{self.octave_offset:+d}]" if self.octave_offset != 0 else ""
        return f"{self.degree}{accidental}{register}/d{self.duration_id}"

    def __str__(self) -> str:
        return repr(self)


class RestToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["rest"] = "rest"
    duration_id: int = Field(ge=MIN_DURATION_ID)

    def to_text(self, *, duration_vocabulary: DurationVocabulary) -> str:
        return f"{REST_SYMBOL}{_duration_text(self.duration_id, duration_vocabulary=duration_vocabulary)}"

    def __repr__(self) -> str:
        return f"{REST_SYMBOL}/d{self.duration_id}"

    def __str__(self) -> str:
        return repr(self)


class HoldToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["hold"] = "hold"
    duration_id: int = Field(ge=MIN_DURATION_ID)

    def to_text(self, *, duration_vocabulary: DurationVocabulary) -> str:
        return f"{HOLD_SYMBOL}{_duration_text(self.duration_id, duration_vocabulary=duration_vocabulary)}"

    def __repr__(self) -> str:
        return f"{HOLD_SYMBOL}/d{self.duration_id}"

    def __str__(self) -> str:
        return repr(self)


class HandToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["hand"] = "hand"
    hand: Hand

    def to_text(self, *, duration_vocabulary: DurationVocabulary) -> str:
        return RIGHT_HAND_SYMBOL if self.hand == Hand.RIGHT else LEFT_HAND_SYMBOL

    def __repr__(self) -> str:
        return f"<{self.hand.value}>"

    def __str__(self) -> str:
        return repr(self)


class JoinWithPreviousToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["join_previous"] = "join_previous"

    def to_text(self, *, duration_vocabulary: DurationVocabulary) -> str:
        return JOIN_WITH_PREVIOUS_SYMBOL

    def __repr__(self) -> str:
        return JOIN_WITH_PREVIOUS_SYMBOL

    def __str__(self) -> str:
        return repr(self)


class BarToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["bar"] = "bar"

    def to_text(self, *, duration_vocabulary: DurationVocabulary) -> str:
        return BAR_SYMBOL

    def __repr__(self) -> str:
        return BAR_SYMBOL

    def __str__(self) -> str:
        return repr(self)


class StartToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["start"] = "start"

    def to_text(self, *, duration_vocabulary: DurationVocabulary) -> str:
        return START_SYMBOL

    def __repr__(self) -> str:
        return START_SYMBOL

    def __str__(self) -> str:
        return repr(self)


class EndToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["end"] = "end"

    def to_text(self, *, duration_vocabulary: DurationVocabulary) -> str:
        return END_SYMBOL

    def __repr__(self) -> str:
        return END_SYMBOL

    def __str__(self) -> str:
        return repr(self)


Token = NoteToken | RestToken | HoldToken | HandToken | JoinWithPreviousToken | BarToken | StartToken | EndToken
