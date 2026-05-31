from __future__ import annotations

import logging
from dataclasses import dataclass

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.distribution import (
    MusicalAuxiliaryTargetSeries,
    musical_auxiliary_bucket_distance_metrics,
    musical_auxiliary_bucket_distribution_metrics,
    musical_auxiliary_target_series,
)
from musak_model.auxiliary.schema import MusicalAuxiliaryTargetIds
from musak_model.auxiliary.targets import (
    bar_musical_auxiliary_target_ids_from_segment,
    musical_auxiliary_target_ids_from_segment,
)
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit

_LOGGER = logging.getLogger(__name__)
_METRIC_PREFIX = "model/split/musical_auxiliary"
_TRAIN_PREFIX = f"{_METRIC_PREFIX}/train"
_VALIDATION_PREFIX = f"{_METRIC_PREFIX}/validation"


@dataclass(frozen=True)
class _SplitAuxiliaryTargets:
    sequence_targets: tuple[MusicalAuxiliaryTargetIds, ...]
    bar_targets: tuple[MusicalAuxiliaryTargetIds, ...]

    def series(self, *, config: MusicalAuxiliaryTargetConfig) -> tuple[MusicalAuxiliaryTargetSeries, ...]:
        return (
            *musical_auxiliary_target_series(self.sequence_targets, config=config, name_prefix=""),
            *musical_auxiliary_target_series(self.bar_targets, config=config, name_prefix="bar_"),
        )


def split_musical_auxiliary_profile_metrics(
    split: IngestionSplit,
    *,
    token_vocabulary: TokenVocabulary,
    target_config: MusicalAuxiliaryTargetConfig,
) -> dict[str, float]:
    _LOGGER.info(
        "Computing train/validation musical auxiliary bucket metrics: train_samples=%s validation_samples=%s",
        len(split.train),
        len(split.validation),
    )
    train_targets = _targets_for_samples(
        split.train,
        token_vocabulary=token_vocabulary,
        target_config=target_config,
    )
    validation_targets = _targets_for_samples(
        split.validation,
        token_vocabulary=token_vocabulary,
        target_config=target_config,
    )
    train_series = train_targets.series(config=target_config)
    validation_series = validation_targets.series(config=target_config)
    metrics = {
        f"{_METRIC_PREFIX}/count/train_samples": float(len(train_targets.sequence_targets)),
        f"{_METRIC_PREFIX}/count/validation_samples": float(len(validation_targets.sequence_targets)),
        f"{_METRIC_PREFIX}/count/train_bars": float(len(train_targets.bar_targets)),
        f"{_METRIC_PREFIX}/count/validation_bars": float(len(validation_targets.bar_targets)),
        **musical_auxiliary_bucket_distribution_metrics(train_series, metric_prefix=_TRAIN_PREFIX),
        **musical_auxiliary_bucket_distribution_metrics(validation_series, metric_prefix=_VALIDATION_PREFIX),
        **musical_auxiliary_bucket_distance_metrics(
            train_series,
            validation_series,
            metric_prefix=_METRIC_PREFIX,
        ),
    }
    _LOGGER.info("Computed %s train/validation musical auxiliary bucket metric(s)", len(metrics))
    return metrics


def _targets_for_samples(
    samples: list[EncodedExercise],
    *,
    token_vocabulary: TokenVocabulary,
    target_config: MusicalAuxiliaryTargetConfig,
) -> _SplitAuxiliaryTargets:
    sequence_targets: list[MusicalAuxiliaryTargetIds] = []
    bar_targets: list[MusicalAuxiliaryTargetIds] = []
    for sample in samples:
        segment = sample.to_segment(token_vocabulary=token_vocabulary)
        sequence_targets.append(
            musical_auxiliary_target_ids_from_segment(
                segment,
                duration_vocabulary=token_vocabulary.duration_vocabulary,
                config=target_config,
            )
        )
        bar_targets.extend(
            bar_musical_auxiliary_target_ids_from_segment(
                segment,
                duration_vocabulary=token_vocabulary.duration_vocabulary,
                config=target_config,
            )
        )

    return _SplitAuxiliaryTargets(
        sequence_targets=tuple(sequence_targets),
        bar_targets=tuple(bar_targets),
    )
