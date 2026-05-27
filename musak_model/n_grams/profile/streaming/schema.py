from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple

from musak_model.n_grams.profile.rhythm.extraction import RhythmCountCounter
from musak_model.tokens.config import TokenizationConfig

type FigureCountCounter = Counter[FigureCountKey]


class FigureCountKey(NamedTuple):
    scale_type: str
    hand: str
    figure_length: int
    figure: str


class FigureGroupKey(NamedTuple):
    scale_type: str
    hand: str
    figure_length: int


class FigureGroupTotals(NamedTuple):
    total: int
    monophonic: int
    chords_only: int
    in_scale: int


type FigureGroupTotalsByKey = dict[FigureGroupKey, FigureGroupTotals]


@dataclass(frozen=True)
class FigureBatchTask:
    batch_index: int
    sample_start_index: int
    encoded_lines: tuple[str, ...]
    tokenization_config: TokenizationConfig
    min_n: int
    max_n: int
    rhythm_min_n: int
    rhythm_max_n: int
    grid_alignment_denominators: tuple[int, ...]
    strong_beat_offsets: tuple[Fraction, ...]


@dataclass(frozen=True)
class FigureBatchResult:
    batch_index: int
    sample_start_index: int
    encoded_sample_count: int
    counts: FigureCountCounter
    rhythm_counts: RhythmCountCounter
    sample_payloads: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FigureStoreSummary:
    encoded_sample_count: int
    profile_group_count: int
    sample_profile_count: int
