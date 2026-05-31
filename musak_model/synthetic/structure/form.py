from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, Field

from musak_model.synthetic.structure.harmony_grammar import ClosingPattern
from musak_shared.elements import HarmonicFunction


class VariationKind(StrEnum):
    FRESH = "fresh"
    SAME = "same"
    VARIANT = "variant"


@dataclass(frozen=True)
class SegmentNode:
    start_bar: int
    bar_span: int
    class_label: int
    variation: VariationKind


@dataclass(frozen=True)
class PhraseNode:
    start_bar: int
    bar_span: int
    closing: ClosingPattern


@dataclass(frozen=True)
class FormTree:
    bar_count: int
    segments: tuple[SegmentNode, ...]
    phrases: tuple[PhraseNode, ...]

    def __post_init__(self) -> None:
        _validate_tiling([(node.start_bar, node.bar_span) for node in self.segments], self.bar_count, "segments")
        _validate_tiling([(node.start_bar, node.bar_span) for node in self.phrases], self.bar_count, "phrases")
        segment_boundaries = {node.start_bar for node in self.segments} | {self.bar_count}
        for phrase in self.phrases:
            phrase_end = phrase.start_bar + phrase.bar_span
            if phrase.start_bar not in segment_boundaries or phrase_end not in segment_boundaries:
                raise ValueError("phrase boundaries must align with segment boundaries")


def _validate_tiling(spans: list[tuple[int, int]], bar_count: int, description: str) -> None:
    expected_start = 0
    for start_bar, bar_span in spans:
        if bar_span <= 0:
            raise ValueError(f"{description} spans must be positive")
        if start_bar != expected_start:
            raise ValueError(f"{description} must tile the piece contiguously")
        expected_start += bar_span

    if expected_start != bar_count:
        raise ValueError(f"{description} must cover all {bar_count} bars")


class WeightedSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bars: int = Field(gt=0)
    weight: float = Field(gt=0)


class ClosingChoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_final: bool
    functions: tuple[HarmonicFunction, ...] = Field(min_length=1)
    weight: float = Field(gt=0)


class FormPrior(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    phrase_lengths: tuple[WeightedSpan, ...] = Field(min_length=1)
    segment_lengths: tuple[WeightedSpan, ...] = Field(min_length=1)
    closings: tuple[ClosingChoice, ...] = Field(min_length=1)
    repeat_probability: float = Field(ge=0.0, le=1.0)
    variation_probability: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class FormSampler:
    prior: FormPrior

    def sample(
        self,
        *,
        bar_count: int,
        rng: Generator,
    ) -> FormTree:
        if bar_count <= 0:
            raise ValueError("bar_count must be positive")

        phrase_spans = _partition(bar_count, self.prior.phrase_lengths, rng)
        phrases: list[PhraseNode] = []
        segments: list[SegmentNode] = []
        next_class_label = 0
        bar = 0
        for phrase_index, phrase_span in enumerate(phrase_spans):
            is_final = phrase_index == len(phrase_spans) - 1
            phrases.append(PhraseNode(start_bar=bar, bar_span=phrase_span, closing=self._sample_closing(is_final, rng)))
            segment_bar = bar
            for segment_span in _partition(phrase_span, self.prior.segment_lengths, rng):
                class_label, variation, next_class_label = self._sample_class(next_class_label, segments, rng)
                segments.append(
                    SegmentNode(
                        start_bar=segment_bar, bar_span=segment_span, class_label=class_label, variation=variation
                    )
                )
                segment_bar += segment_span
            bar += phrase_span

        return FormTree(bar_count=bar_count, segments=tuple(segments), phrases=tuple(phrases))

    def _sample_closing(self, is_final: bool, rng: Generator) -> ClosingPattern:
        choices = tuple(choice for choice in self.prior.closings if choice.is_final == is_final)
        if not choices:
            raise ValueError(f"form prior has no closing for is_final={is_final}")

        choice = choices[_weighted_index([choice.weight for choice in choices], rng)]
        return ClosingPattern(choice.functions)

    def _sample_class(
        self,
        next_class_label: int,
        segments: Sequence[SegmentNode],
        rng: Generator,
    ) -> tuple[int, VariationKind, int]:
        if not segments or rng.random() >= self.prior.repeat_probability:
            return next_class_label, VariationKind.FRESH, next_class_label + 1

        existing_labels = sorted({segment.class_label for segment in segments})
        class_label = existing_labels[int(rng.integers(0, len(existing_labels)))]
        variation = VariationKind.VARIANT if rng.random() < self.prior.variation_probability else VariationKind.SAME
        return class_label, variation, next_class_label


def _partition(
    total_bars: int,
    choices: Sequence[WeightedSpan],
    rng: Generator,
) -> list[int]:
    weights = [choice.weight for choice in choices]
    spans: list[int] = []
    remaining = total_bars
    while remaining > 0:
        span = min(choices[_weighted_index(weights, rng)].bars, remaining)
        spans.append(span)
        remaining -= span

    return spans


def _weighted_index(
    weights: Sequence[float],
    rng: Generator,
) -> int:
    probabilities = np.array(weights, dtype=np.float64)
    return int(rng.choice(len(weights), p=probabilities / probabilities.sum()))
