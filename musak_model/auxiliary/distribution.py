from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.schema import MusicalAuxiliaryTargetIds

type _TargetIdGetter = Callable[[MusicalAuxiliaryTargetIds], int]
type _ClassCountGetter = Callable[[MusicalAuxiliaryTargetConfig], int]


@dataclass(frozen=True)
class MusicalAuxiliaryTargetSeries:
    name: str
    target_ids: tuple[int, ...]
    class_count: int


@dataclass(frozen=True)
class _MusicalAuxiliaryTargetSpec:
    name: str
    target_id: _TargetIdGetter
    class_count: _ClassCountGetter


_MUSICAL_AUXILIARY_TARGET_SPECS: Final[tuple[_MusicalAuxiliaryTargetSpec, ...]] = (
    _MusicalAuxiliaryTargetSpec(
        name="note_density",
        target_id=lambda target: target.note_density_id,
        class_count=lambda config: config.note_density_class_count,
    ),
    _MusicalAuxiliaryTargetSpec(
        name="rhythmic_diversity",
        target_id=lambda target: target.rhythmic_diversity_id,
        class_count=lambda config: config.rhythmic_diversity_class_count,
    ),
    _MusicalAuxiliaryTargetSpec(
        name="voice_independence",
        target_id=lambda target: target.voice_independence_id,
        class_count=lambda config: config.voice_independence_class_count,
    ),
    _MusicalAuxiliaryTargetSpec(
        name="uses_accidentals",
        target_id=lambda target: target.uses_accidentals_id,
        class_count=lambda config: config.uses_accidentals_class_count,
    ),
    _MusicalAuxiliaryTargetSpec(
        name="dotted_duration",
        target_id=lambda target: target.dotted_duration_id,
        class_count=lambda config: config.dotted_duration_class_count,
    ),
    _MusicalAuxiliaryTargetSpec(
        name="hand_span",
        target_id=lambda target: target.hand_span_id,
        class_count=lambda config: config.hand_span_class_count,
    ),
)


def musical_auxiliary_target_series(
    targets: Sequence[MusicalAuxiliaryTargetIds],
    *,
    config: MusicalAuxiliaryTargetConfig,
    name_prefix: str,
) -> tuple[MusicalAuxiliaryTargetSeries, ...]:
    return tuple(
        MusicalAuxiliaryTargetSeries(
            name=f"{name_prefix}{spec.name}",
            target_ids=tuple(spec.target_id(target) for target in targets),
            class_count=spec.class_count(config),
        )
        for spec in _MUSICAL_AUXILIARY_TARGET_SPECS
    )


def musical_auxiliary_bucket_distribution_metrics(
    series_values: Sequence[MusicalAuxiliaryTargetSeries],
    *,
    metric_prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for series in series_values:
        if not series.target_ids:
            continue

        total_count = len(series.target_ids)
        counts = Counter(series.target_ids)
        metrics[f"{metric_prefix}/mean/{series.name}_bucket_id"] = sum(series.target_ids) / total_count
        for bucket_id in range(series.class_count):
            metrics[f"{metric_prefix}/rate/{series.name}_bucket_{bucket_id}"] = counts.get(bucket_id, 0) / total_count

    return metrics


def musical_auxiliary_bucket_distance_metrics(
    reference_series: Sequence[MusicalAuxiliaryTargetSeries],
    comparison_series: Sequence[MusicalAuxiliaryTargetSeries],
    *,
    metric_prefix: str,
) -> dict[str, float]:
    comparison_by_name = {series.name: series for series in comparison_series}
    metrics: dict[str, float] = {}
    comparable_count = 0
    for reference in reference_series:
        comparison = comparison_by_name[reference.name]
        if not reference.target_ids or not comparison.target_ids:
            continue

        if reference.class_count != comparison.class_count:
            raise ValueError(f"class count mismatch for auxiliary target series {reference.name!r}")

        comparable_count += 1
        metrics[f"{metric_prefix}/mean/{reference.name}_total_variation_distance"] = _total_variation_distance(
            reference, comparison
        )

    metrics[f"{metric_prefix}/count/comparable_distributions"] = float(comparable_count)
    return metrics


def _total_variation_distance(
    reference: MusicalAuxiliaryTargetSeries,
    comparison: MusicalAuxiliaryTargetSeries,
) -> float:
    reference_rates = _bucket_rates(reference)
    comparison_rates = _bucket_rates(comparison)
    return 0.5 * sum(
        abs(reference_rates[bucket_id] - comparison_rates[bucket_id]) for bucket_id in range(reference.class_count)
    )


def _bucket_rates(series: MusicalAuxiliaryTargetSeries) -> tuple[float, ...]:
    if not series.target_ids:
        raise ValueError("cannot calculate bucket rates for an empty target series")

    counts = Counter(series.target_ids)
    total_count = len(series.target_ids)
    return tuple(counts.get(bucket_id, 0) / total_count for bucket_id in range(series.class_count))
