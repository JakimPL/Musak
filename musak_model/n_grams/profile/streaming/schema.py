from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple

from musak_model.n_grams.profile.chord.schema import ChordDecodeSpec, ChordStatistics
from musak_model.n_grams.profile.register.schema import RegisterStatistics
from musak_model.n_grams.profile.rhythm.schema import RhythmCountCounter
from musak_model.tokens.config import TokenizationConfig

type FigureCountCounter = Counter[FigureCountKey]


class FigureCountKey(NamedTuple):
    scale_type: str
    hand: str
    figure_length: int
    figure: str
    anchor_degree: int
    anchor_accidental: int
    anchor_octave: int
    base_duration: str
    bar_relative_onset: str
    time_signature: str


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
    register_arch_basis_count: int
    chord_decode: ChordDecodeSpec | None = None


@dataclass(frozen=True)
class FigureBatchResult:
    batch_index: int
    sample_start_index: int
    encoded_sample_count: int
    counts: FigureCountCounter
    rhythm_counts: RhythmCountCounter
    register_statistics: RegisterStatistics
    chord_statistics: ChordStatistics
    sample_payloads: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FigureStoreSummary:
    encoded_sample_count: int
    profile_group_count: int
    sample_profile_count: int
