from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, NamedTuple

from musak_model.synthetic.fitting.form.analysis import AnalyzedPiece
from musak_model.synthetic.fitting.form.cadence import Cadence
from musak_model.synthetic.fitting.form.repetition import RepetitionAnalysis
from musak_shared.elements import HarmonicFunction

_CLOSING_SEPARATOR: Final[str] = ">"


class PhraseLengthKey(NamedTuple):
    scale_type: str
    phrase_length_bars: int


class SegmentLengthKey(NamedTuple):
    scale_type: str
    segment_length_bars: int


class ClosingKey(NamedTuple):
    scale_type: str
    is_final: bool
    functions: str


class HistogramKey(NamedTuple):
    scale_type: str
    bucket: int


type PhraseLengthCounts = Counter[PhraseLengthKey]
type SegmentLengthCounts = Counter[SegmentLengthKey]
type ClosingCounts = Counter[ClosingKey]
type HistogramCounts = Counter[HistogramKey]


@dataclass(frozen=True)
class FormStatistics:
    phrase_length_counts: PhraseLengthCounts = field(default_factory=Counter)
    segment_length_counts: SegmentLengthCounts = field(default_factory=Counter)
    closing_counts: ClosingCounts = field(default_factory=Counter)
    similarity_histogram: HistogramCounts = field(default_factory=Counter)
    best_match_histogram: HistogramCounts = field(default_factory=Counter)


def accumulate_piece(
    statistics: FormStatistics,
    *,
    piece: AnalyzedPiece,
    cadences: Sequence[Cadence],
    repetition: RepetitionAnalysis | None,
    similarity_bucket_count: int,
) -> None:
    scale_type = piece.scale_type.value
    previous_end_bar = 0
    for cadence in cadences:
        phrase_length = cadence.end_bar - previous_end_bar
        previous_end_bar = cadence.end_bar
        if phrase_length > 0:
            statistics.phrase_length_counts[PhraseLengthKey(scale_type, phrase_length)] += 1

        functions = closing_functions_to_key(cadence.closing.functions)
        statistics.closing_counts[ClosingKey(scale_type, cadence.is_final, functions)] += 1

    if repetition is not None:
        statistics.segment_length_counts[SegmentLengthKey(scale_type, repetition.segment_length)] += 1
        for similarity in repetition.pairwise_similarities:
            bucket = similarity_bucket(similarity, similarity_bucket_count)
            statistics.similarity_histogram[HistogramKey(scale_type, bucket)] += 1

        for similarity in repetition.best_match_similarities:
            bucket = similarity_bucket(similarity, similarity_bucket_count)
            statistics.best_match_histogram[HistogramKey(scale_type, bucket)] += 1


def closing_functions_to_key(functions: tuple[HarmonicFunction, ...]) -> str:
    return _CLOSING_SEPARATOR.join(function.value for function in functions)


def closing_functions_from_key(key: str) -> tuple[HarmonicFunction, ...]:
    return tuple(HarmonicFunction(value) for value in key.split(_CLOSING_SEPARATOR))


def similarity_bucket(value: float, bucket_count: int) -> int:
    return min(bucket_count - 1, max(0, int(value * bucket_count)))


def bucket_center(bucket: int, bucket_count: int) -> float:
    return (bucket + 0.5) / bucket_count
