from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from musak_model.paths import FORM_FITTING_CONFIG_PATH
from musak_model.synthetic.fitting.form.cadence import CadenceDetectionConfig
from musak_model.synthetic.fitting.form.repetition import RepetitionConfig
from musak_model.synthetic.fitting.form.statistics import (
    FormStatistics,
    HistogramCounts,
    bucket_center,
    closing_functions_from_key,
)
from musak_model.synthetic.structure.form import ClosingChoice, FormPrior, WeightedSpan
from musak_model.tokens.schema import ScaleType
from musak_shared.elements import HarmonicFunction
from musak_shared.files import load_yaml_config

_KeyType = TypeVar("_KeyType")


class FormFittingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cadence: CadenceDetectionConfig
    repetition: RepetitionConfig
    smoothing_pseudo_count: float = Field(ge=0)
    minimum_observation_count: int = Field(gt=0)
    same_similarity_threshold: float = Field(ge=0, le=1)
    variation_similarity_threshold: float = Field(ge=0, le=1)
    fallback_prior: FormPrior

    @classmethod
    def load(cls, path: Path = FORM_FITTING_CONFIG_PATH) -> FormFittingConfig:
        return cls.model_validate(load_yaml_config(path))


def fit_form_priors(statistics: FormStatistics, *, config: FormFittingConfig) -> dict[ScaleType, FormPrior]:
    return {
        ScaleType(scale_type): _fit_scale_prior(statistics, scale_type=scale_type, config=config)
        for scale_type in sorted(_scale_types(statistics))
    }


def _fit_scale_prior(statistics: FormStatistics, *, scale_type: str, config: FormFittingConfig) -> FormPrior:
    closing_counts = {
        (key.is_final, key.functions): count
        for key, count in statistics.closing_counts.items()
        if key.scale_type == scale_type
    }
    if sum(closing_counts.values()) < config.minimum_observation_count:
        return config.fallback_prior

    pseudo = config.smoothing_pseudo_count
    fallback = config.fallback_prior
    bucket_count = config.repetition.similarity_bucket_count
    variation_threshold, same_threshold = _calibrate_thresholds(
        _histogram(statistics.similarity_histogram, scale_type, bucket_count), config=config
    )
    repeat_probability, variation_probability = _repeat_rates(
        _histogram(statistics.best_match_histogram, scale_type, bucket_count),
        variation_threshold=variation_threshold,
        same_threshold=same_threshold,
        bucket_count=bucket_count,
        fallback_repeat=fallback.repeat_probability,
        fallback_variation=fallback.variation_probability,
        pseudo=pseudo,
    )
    phrase_lengths = _weighted_spans(_phrase_corpus(statistics, scale_type), fallback.phrase_lengths, pseudo)
    segment_lengths = _weighted_spans(_segment_corpus(statistics, scale_type), fallback.segment_lengths, pseudo)
    closings = _closing_choices(closing_counts, fallback.closings, pseudo)
    return FormPrior(
        phrase_lengths=phrase_lengths or fallback.phrase_lengths,
        segment_lengths=segment_lengths or fallback.segment_lengths,
        closings=closings or fallback.closings,
        repeat_probability=repeat_probability,
        variation_probability=variation_probability,
    )


def _scale_types(statistics: FormStatistics) -> set[str]:
    counters = (
        statistics.phrase_length_counts,
        statistics.segment_length_counts,
        statistics.closing_counts,
        statistics.similarity_histogram,
        statistics.best_match_histogram,
    )
    return {key.scale_type for counter in counters for key in counter}


def _phrase_corpus(statistics: FormStatistics, scale_type: str) -> dict[int, int]:
    return {
        key.phrase_length_bars: count
        for key, count in statistics.phrase_length_counts.items()
        if key.scale_type == scale_type
    }


def _segment_corpus(statistics: FormStatistics, scale_type: str) -> dict[int, int]:
    return {
        key.segment_length_bars: count
        for key, count in statistics.segment_length_counts.items()
        if key.scale_type == scale_type
    }


def _histogram(counts: HistogramCounts, scale_type: str, bucket_count: int) -> list[int]:
    histogram = [0] * bucket_count
    for key, count in counts.items():
        if key.scale_type == scale_type and 0 <= key.bucket < bucket_count:
            histogram[key.bucket] += count

    return histogram


def _calibrate_thresholds(similarity_histogram: list[int], *, config: FormFittingConfig) -> tuple[float, float]:
    bucket_count = config.repetition.similarity_bucket_count
    if sum(similarity_histogram) < config.minimum_observation_count:
        return config.variation_similarity_threshold, config.same_similarity_threshold

    variation_bucket = _otsu_bucket(similarity_histogram)
    if variation_bucket is None:
        return config.variation_similarity_threshold, config.same_similarity_threshold

    variation_threshold = (variation_bucket + 1) / bucket_count
    upper = similarity_histogram[variation_bucket + 1 :]
    same_bucket = _otsu_bucket(upper) if sum(upper) >= config.minimum_observation_count else None
    same_threshold = (
        config.same_similarity_threshold
        if same_bucket is None
        else (variation_bucket + 1 + same_bucket + 1) / bucket_count
    )
    return variation_threshold, max(variation_threshold, same_threshold)


def _otsu_bucket(histogram: list[int]) -> int | None:
    total = sum(histogram)
    if total == 0:
        return None

    weighted_total = sum(index * count for index, count in enumerate(histogram))
    background_weight = 0
    background_weighted = 0
    best_variance = -1.0
    best_bucket: int | None = None
    for bucket, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue

        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break

        background_weighted += bucket * count
        background_mean = background_weighted / background_weight
        foreground_mean = (weighted_total - background_weighted) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_bucket = bucket

    return best_bucket


def _repeat_rates(
    best_match_histogram: list[int],
    *,
    variation_threshold: float,
    same_threshold: float,
    bucket_count: int,
    fallback_repeat: float,
    fallback_variation: float,
    pseudo: float,
) -> tuple[float, float]:
    total = sum(best_match_histogram)
    repeat_mass = 0
    variant_mass = 0
    for bucket, count in enumerate(best_match_histogram):
        center = bucket_center(bucket, bucket_count)
        if center >= variation_threshold:
            repeat_mass += count
            if center < same_threshold:
                variant_mass += count

    return (
        _smoothed_probability(repeat_mass, total, fallback_repeat, pseudo),
        _smoothed_probability(variant_mass, repeat_mass, fallback_variation, pseudo),
    )


def _weighted_spans(
    corpus_counts: Mapping[int, int],
    fallback_spans: tuple[WeightedSpan, ...],
    pseudo: float,
) -> tuple[WeightedSpan, ...]:
    fallback_probabilities = _normalized({span.bars: span.weight for span in fallback_spans})
    spans: list[WeightedSpan] = []
    for bars in sorted(set(corpus_counts) | set(fallback_probabilities)):
        weight = _smoothed_weight(corpus_counts.get(bars, 0), fallback_probabilities.get(bars, 0.0), pseudo)
        if weight > 0:
            spans.append(WeightedSpan(bars=bars, weight=weight))

    return tuple(spans)


def _closing_choices(
    corpus_counts: Mapping[tuple[bool, str], int],
    fallback_choices: tuple[ClosingChoice, ...],
    pseudo: float,
) -> tuple[ClosingChoice, ...]:
    choices: list[ClosingChoice] = []
    for is_final in (False, True):
        corpus_group = _corpus_closing_group(corpus_counts, is_final)
        fallback_probabilities = _normalized(
            {choice.functions: choice.weight for choice in fallback_choices if choice.is_final == is_final}
        )
        for functions in set(corpus_group) | set(fallback_probabilities):
            weight = _smoothed_weight(
                corpus_group.get(functions, 0.0), fallback_probabilities.get(functions, 0.0), pseudo
            )
            if weight > 0:
                choices.append(
                    ClosingChoice(
                        is_final=is_final,
                        functions=functions,
                        weight=weight,
                    )
                )

    return tuple(choices)


def _corpus_closing_group(
    corpus_counts: Mapping[tuple[bool, str], int],
    is_final: bool,
) -> dict[tuple[HarmonicFunction, ...], float]:
    group: dict[tuple[HarmonicFunction, ...], float] = {}
    for (final, functions), count in corpus_counts.items():
        if final == is_final:
            pattern = closing_functions_from_key(functions)
            group[pattern] = group.get(pattern, 0.0) + count

    return group


def _normalized(weights: Mapping[_KeyType, float]) -> dict[_KeyType, float]:
    total = sum(weights.values())
    if total <= 0:
        return {}

    return {key: weight / total for key, weight in weights.items()}


def _smoothed_weight(corpus_count: float, fallback_probability: float, pseudo: float) -> float:
    return corpus_count + pseudo * fallback_probability


def _smoothed_probability(numerator: float, denominator: float, fallback: float, pseudo: float) -> float:
    if denominator + pseudo <= 0:
        return _clamp(fallback)

    return _clamp((numerator + pseudo * fallback) / (denominator + pseudo))


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
