from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.schema import MusicalAuxiliaryTargetIds
from musak_model.auxiliary.targets import (
    bar_musical_auxiliary_target_ids_from_segment,
    musical_auxiliary_target_ids_from_segment,
)
from musak_model.evaluation.generation.protocols import GenerationEvaluationOptions
from musak_model.evaluation.generation.sampling import segment_from_tokens
from musak_model.evaluation.generation.schema import GenerationSample
from musak_model.tokens.duration import DurationVocabulary

_METRIC_PREFIX: Final[str] = "generation/musical_auxiliary"


@dataclass(frozen=True)
class _BucketSeries:
    name: str
    target_ids: tuple[int, ...]
    class_count: int


def musical_auxiliary_bucket_metrics(
    *,
    samples: list[GenerationSample],
    config: GenerationEvaluationOptions,
    target_config: MusicalAuxiliaryTargetConfig,
    duration_vocabulary: DurationVocabulary,
) -> dict[str, float]:
    sample_targets: list[MusicalAuxiliaryTargetIds] = []
    bar_targets: list[MusicalAuxiliaryTargetIds] = []
    skipped_decode_error_count = 0
    for sample in samples:
        if sample.decode_error is not None:
            skipped_decode_error_count += 1
            continue

        segment = segment_from_tokens(sample.tokens, config=config)
        sample_targets.append(
            musical_auxiliary_target_ids_from_segment(
                segment,
                duration_vocabulary=duration_vocabulary,
                config=target_config,
            )
        )
        bar_targets.extend(
            bar_musical_auxiliary_target_ids_from_segment(
                segment,
                duration_vocabulary=duration_vocabulary,
                config=target_config,
            )
        )

    metrics = {
        f"{_METRIC_PREFIX}/count/samples": float(len(sample_targets)),
        f"{_METRIC_PREFIX}/count/skipped_decode_errors": float(skipped_decode_error_count),
        f"{_METRIC_PREFIX}/count/bars": float(len(bar_targets)),
    }
    metrics.update(_bucket_metrics(_sample_bucket_series(sample_targets, config=target_config)))
    metrics.update(_bucket_metrics(_bar_bucket_series(bar_targets, config=target_config)))
    return metrics


def _sample_bucket_series(
    targets: list[MusicalAuxiliaryTargetIds],
    *,
    config: MusicalAuxiliaryTargetConfig,
) -> tuple[_BucketSeries, ...]:
    return (
        _BucketSeries(
            name="note_density",
            target_ids=tuple(target.note_density_id for target in targets),
            class_count=config.note_density_class_count,
        ),
        _BucketSeries(
            name="rhythmic_diversity",
            target_ids=tuple(target.rhythmic_diversity_id for target in targets),
            class_count=config.rhythmic_diversity_class_count,
        ),
        _BucketSeries(
            name="voice_independence",
            target_ids=tuple(target.voice_independence_id for target in targets),
            class_count=config.voice_independence_class_count,
        ),
        _BucketSeries(
            name="uses_accidentals",
            target_ids=tuple(target.uses_accidentals_id for target in targets),
            class_count=config.uses_accidentals_class_count,
        ),
        _BucketSeries(
            name="dotted_duration",
            target_ids=tuple(target.dotted_duration_id for target in targets),
            class_count=config.dotted_duration_class_count,
        ),
        _BucketSeries(
            name="hand_span",
            target_ids=tuple(target.hand_span_id for target in targets),
            class_count=config.hand_span_class_count,
        ),
    )


def _bar_bucket_series(
    targets: list[MusicalAuxiliaryTargetIds],
    *,
    config: MusicalAuxiliaryTargetConfig,
) -> tuple[_BucketSeries, ...]:
    return tuple(
        _BucketSeries(
            name=f"bar_{series.name}",
            target_ids=series.target_ids,
            class_count=series.class_count,
        )
        for series in _sample_bucket_series(targets, config=config)
    )


def _bucket_metrics(series_values: tuple[_BucketSeries, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for series in series_values:
        if not series.target_ids:
            continue

        total_count = len(series.target_ids)
        counts = Counter(series.target_ids)
        metrics[f"{_METRIC_PREFIX}/mean/{series.name}_bucket_id"] = sum(series.target_ids) / total_count
        for bucket_id in range(series.class_count):
            metrics[f"{_METRIC_PREFIX}/rate/{series.name}_bucket_{bucket_id}"] = counts.get(bucket_id, 0) / total_count

    return metrics
