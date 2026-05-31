from __future__ import annotations

from typing import Final

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.distribution import (
    musical_auxiliary_bucket_distribution_metrics,
    musical_auxiliary_target_series,
)
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
    metrics.update(
        musical_auxiliary_bucket_distribution_metrics(
            musical_auxiliary_target_series(sample_targets, config=target_config, name_prefix=""),
            metric_prefix=_METRIC_PREFIX,
        )
    )
    metrics.update(
        musical_auxiliary_bucket_distribution_metrics(
            musical_auxiliary_target_series(bar_targets, config=target_config, name_prefix="bar_"),
            metric_prefix=_METRIC_PREFIX,
        )
    )
    return metrics
