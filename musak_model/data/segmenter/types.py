from fractions import Fraction
from typing import NamedTuple

from musak_model.data.schema import SegmentIneligibilityReason
from musak_model.tokens.schema import Hand, Token


class TimedTokenGroup(NamedTuple):
    bar_index: int
    offset: Fraction
    hand: Hand
    tokens: list[Token]


class BarTokenization(NamedTuple):
    tokens: list[Token]
    ineligibility_reasons: frozenset[SegmentIneligibilityReason] = frozenset()


class TieState(NamedTuple):
    midi_pitches: tuple[int, ...]


class BarNormalization(NamedTuple):
    groups: list[TimedTokenGroup]
    tie_state: TieState | None
