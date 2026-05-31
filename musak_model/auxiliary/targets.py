from __future__ import annotations

import torch
from torch import Tensor

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.features import (
    MusicalAuxiliaryFeatures,
    bar_musical_auxiliary_features_from_segment,
    musical_auxiliary_features_from_segment,
)
from musak_model.auxiliary.schema import (
    MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
    MusicalAuxiliaryTargetIds,
    MusicalAuxiliaryTargetTensors,
    MusicalBarAuxiliaryTargetTensors,
)
from musak_model.data.schema import DifficultyFeatures, Segment
from musak_model.tokens.duration import DurationVocabulary


def musical_auxiliary_target_ids_from_difficulty_features(
    features: DifficultyFeatures | None,
    *,
    config: MusicalAuxiliaryTargetConfig,
) -> MusicalAuxiliaryTargetIds:
    if features is None:
        return _ignored_target_ids()

    return musical_auxiliary_target_ids_from_features(
        MusicalAuxiliaryFeatures(
            note_density=features.notes_per_beat,
            rhythmic_diversity=features.rhythmic_diversity,
            voice_independence=features.voice_independence,
            uses_accidentals=features.has_accidentals,
            dotted_duration=features.has_dotted_notes,
            hand_span=max(features.max_right_hand_span_semitones, features.max_left_hand_span_semitones),
        ),
        config=config,
    )


def musical_auxiliary_target_ids_from_features(
    features: MusicalAuxiliaryFeatures,
    *,
    config: MusicalAuxiliaryTargetConfig,
) -> MusicalAuxiliaryTargetIds:
    return MusicalAuxiliaryTargetIds(
        note_density_id=_float_bucket_id(features.note_density, config.note_density_bucket_boundaries),
        rhythmic_diversity_id=_float_bucket_id(
            features.rhythmic_diversity,
            config.rhythmic_diversity_bucket_boundaries,
        ),
        voice_independence_id=_float_bucket_id(
            features.voice_independence,
            config.voice_independence_bucket_boundaries,
        ),
        uses_accidentals_id=int(features.uses_accidentals),
        dotted_duration_id=int(features.dotted_duration),
        hand_span_id=_integer_bucket_id(features.hand_span, config.hand_span_bucket_boundaries),
    )


def musical_auxiliary_target_ids_from_segment(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    config: MusicalAuxiliaryTargetConfig,
) -> MusicalAuxiliaryTargetIds:
    return musical_auxiliary_target_ids_from_features(
        musical_auxiliary_features_from_segment(segment, duration_vocabulary=duration_vocabulary),
        config=config,
    )


def bar_musical_auxiliary_target_ids_from_segment(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    config: MusicalAuxiliaryTargetConfig,
) -> tuple[MusicalAuxiliaryTargetIds, ...]:
    return tuple(
        musical_auxiliary_target_ids_from_features(features, config=config)
        for features in bar_musical_auxiliary_features_from_segment(
            segment,
            duration_vocabulary=duration_vocabulary,
        )
    )


def musical_auxiliary_target_tensors_from_ids(
    target_ids: MusicalAuxiliaryTargetIds,
    *,
    bar_target_ids: tuple[MusicalAuxiliaryTargetIds, ...],
) -> MusicalAuxiliaryTargetTensors:
    return MusicalAuxiliaryTargetTensors(
        note_density_ids=_target_tensor(target_ids.note_density_id),
        rhythmic_diversity_ids=_target_tensor(target_ids.rhythmic_diversity_id),
        voice_independence_ids=_target_tensor(target_ids.voice_independence_id),
        uses_accidentals_ids=_target_tensor(target_ids.uses_accidentals_id),
        dotted_duration_ids=_target_tensor(target_ids.dotted_duration_id),
        hand_span_ids=_target_tensor(target_ids.hand_span_id),
        bar_targets=bar_musical_auxiliary_target_tensors_from_ids(bar_target_ids),
    )


def bar_musical_auxiliary_target_tensors_from_ids(
    target_ids: tuple[MusicalAuxiliaryTargetIds, ...],
) -> MusicalBarAuxiliaryTargetTensors:
    return MusicalBarAuxiliaryTargetTensors(
        note_density_ids=_target_tensor_series([target.note_density_id for target in target_ids]),
        rhythmic_diversity_ids=_target_tensor_series([target.rhythmic_diversity_id for target in target_ids]),
        voice_independence_ids=_target_tensor_series([target.voice_independence_id for target in target_ids]),
        uses_accidentals_ids=_target_tensor_series([target.uses_accidentals_id for target in target_ids]),
        dotted_duration_ids=_target_tensor_series([target.dotted_duration_id for target in target_ids]),
        hand_span_ids=_target_tensor_series([target.hand_span_id for target in target_ids]),
    )


def stack_musical_auxiliary_targets(
    targets: list[MusicalAuxiliaryTargetTensors],
) -> MusicalAuxiliaryTargetTensors:
    return MusicalAuxiliaryTargetTensors(
        note_density_ids=_stack_target_tensors([target.note_density_ids for target in targets]),
        rhythmic_diversity_ids=_stack_target_tensors([target.rhythmic_diversity_ids for target in targets]),
        voice_independence_ids=_stack_target_tensors([target.voice_independence_ids for target in targets]),
        uses_accidentals_ids=_stack_target_tensors([target.uses_accidentals_ids for target in targets]),
        dotted_duration_ids=_stack_target_tensors([target.dotted_duration_ids for target in targets]),
        hand_span_ids=_stack_target_tensors([target.hand_span_ids for target in targets]),
        bar_targets=stack_musical_bar_auxiliary_targets([target.bar_targets for target in targets]),
    )


def stack_musical_bar_auxiliary_targets(
    targets: list[MusicalBarAuxiliaryTargetTensors],
) -> MusicalBarAuxiliaryTargetTensors:
    return MusicalBarAuxiliaryTargetTensors(
        note_density_ids=_pad_bar_target_tensors([target.note_density_ids for target in targets]),
        rhythmic_diversity_ids=_pad_bar_target_tensors([target.rhythmic_diversity_ids for target in targets]),
        voice_independence_ids=_pad_bar_target_tensors([target.voice_independence_ids for target in targets]),
        uses_accidentals_ids=_pad_bar_target_tensors([target.uses_accidentals_ids for target in targets]),
        dotted_duration_ids=_pad_bar_target_tensors([target.dotted_duration_ids for target in targets]),
        hand_span_ids=_pad_bar_target_tensors([target.hand_span_ids for target in targets]),
    )


def _ignored_target_ids() -> MusicalAuxiliaryTargetIds:
    return MusicalAuxiliaryTargetIds(
        note_density_id=MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
        rhythmic_diversity_id=MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
        voice_independence_id=MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
        uses_accidentals_id=MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
        dotted_duration_id=MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
        hand_span_id=MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
    )


def _float_bucket_id(value: float, boundaries: tuple[float, ...]) -> int:
    for bucket_id, upper_bound in enumerate(boundaries):
        if value < upper_bound:
            return bucket_id

    return len(boundaries)


def _integer_bucket_id(value: int, boundaries: tuple[int, ...]) -> int:
    for bucket_id, upper_bound in enumerate(boundaries):
        if value < upper_bound:
            return bucket_id

    return len(boundaries)


def _target_tensor(target_id: int) -> Tensor:
    return torch.tensor(target_id, dtype=torch.long)


def _target_tensor_series(target_ids: list[int]) -> Tensor:
    return torch.tensor(target_ids, dtype=torch.long)


def _stack_target_tensors(targets: list[Tensor]) -> Tensor:
    if not targets:
        raise ValueError("cannot stack empty musical auxiliary targets")

    return torch.stack(targets)


def _pad_bar_target_tensors(targets: list[Tensor]) -> Tensor:
    if not targets:
        raise ValueError("cannot stack empty musical auxiliary bar targets")

    max_length = max(target.size(0) for target in targets)
    output = torch.full(
        (len(targets), max_length),
        MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
        dtype=torch.long,
    )
    for row_index, target in enumerate(targets):
        if target.ndim != 1:
            raise ValueError(f"bar auxiliary target tensors must be 1D, got {target.ndim}D")

        output[row_index, : target.size(0)] = target

    return output
