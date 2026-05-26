from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

from musak_model.tokens.config import TokenizationConfig

type FigureCountKey = tuple[str, str, int, str]
type FigureCountCounter = Counter[FigureCountKey]
type FigureGroupKey = tuple[str, str, int]


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


@dataclass(frozen=True)
class FigureBatchResult:
    batch_index: int
    sample_start_index: int
    encoded_sample_count: int
    counts: FigureCountCounter
    sample_payloads: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FigureStoreSummary:
    encoded_sample_count: int
    profile_group_count: int
    sample_profile_count: int
