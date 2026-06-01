from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.metrics.stats import total_variation_distance
from musak_model.synthetic.fitting.form.analysis import AnalyzedPiece


class RepetitionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_length_candidates: tuple[int, ...] = Field(min_length=1)
    similarity_bucket_count: int = Field(gt=0)


@dataclass(frozen=True)
class RepetitionAnalysis:
    segment_length: int
    pairwise_similarities: tuple[float, ...]
    best_match_similarities: tuple[float, ...]


def analyze_repetition(piece: AnalyzedPiece, *, config: RepetitionConfig) -> RepetitionAnalysis | None:
    best_analysis: RepetitionAnalysis | None = None
    best_key: tuple[float, int] | None = None
    for segment_length in config.segment_length_candidates:
        if not 0 < segment_length <= piece.bar_count:
            continue

        segments = _segment_figures(piece.bar_figures, segment_length)
        if len(segments) < 2:
            continue

        best_matches = _best_earlier_matches(segments)
        if not best_matches:
            continue

        key = (sum(best_matches) / len(best_matches), segment_length)
        if best_key is None or key > best_key:
            best_key = key
            best_analysis = RepetitionAnalysis(
                segment_length=segment_length,
                pairwise_similarities=_all_pairwise(segments),
                best_match_similarities=tuple(best_matches),
            )

    return best_analysis


def _segment_figures(
    bar_figures: tuple[Counter[FigureNGram], ...],
    segment_length: int,
) -> list[Counter[FigureNGram]]:
    segments: list[Counter[FigureNGram]] = []
    start = 0
    while start < len(bar_figures):
        end = min(start + segment_length, len(bar_figures))
        segment: Counter[FigureNGram] = Counter()
        for bar_index in range(start, end):
            segment.update(bar_figures[bar_index])

        segments.append(segment)
        start = end

    return segments


def _best_earlier_matches(segments: list[Counter[FigureNGram]]) -> list[float]:
    best_matches: list[float] = []
    for later in range(1, len(segments)):
        if not segments[later]:
            continue

        similarities = [
            _similarity(segments[earlier], segments[later]) for earlier in range(later) if segments[earlier]
        ]
        if similarities:
            best_matches.append(max(similarities))

    return best_matches


def _all_pairwise(segments: list[Counter[FigureNGram]]) -> tuple[float, ...]:
    return tuple(
        _similarity(segments[earlier], segments[later])
        for earlier in range(len(segments))
        for later in range(earlier + 1, len(segments))
        if segments[earlier] and segments[later]
    )


def _similarity(left: Counter[FigureNGram], right: Counter[FigureNGram]) -> float:
    return 1.0 - total_variation_distance(left, right)
