import logging
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor
from torch.utils.data import Dataset

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.targets import (
    bar_musical_auxiliary_target_ids_from_segment,
    musical_auxiliary_target_ids_from_difficulty_features,
    musical_auxiliary_target_tensors_from_ids,
)
from musak_model.conditioning.harmony.alignment import harmonic_plan_tensors_from_decoder_coordinates
from musak_model.conditioning.harmony.extraction import harmonic_plan_windows_from_segment
from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors
from musak_model.conditioning.structural.features import extract_structural_control_features
from musak_model.conditioning.structural.vocabulary import StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.data.schema import Segment
from musak_model.generation.constraints import GenerationConstraints
from musak_model.generation.coordinates import DecoderInputCoordinates, decoder_input_coordinates_from_token_ids
from musak_model.harmony.decoding import ChordDecoderConfig, ViterbiChordDecoder
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.duration import duration_tick_denominator
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.conditioning import difficulty_level_to_id, scale_type_to_id, time_signature_to_id
from musak_model.training.config import TrainingConditioningConfig
from musak_model.training.dataset.factorized import token_attribute_targets_from_token_ids
from musak_model.training.dataset.schema import TrainingExample
from musak_model.training.ingestion.schema import EncodedExercise

_START_BAR_POSITION: Final[int] = 0
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _HarmonicPlanDatasetContext:
    decoder: ViterbiChordDecoder
    vocabulary: ChordVocabularyConfig


class EncodedExerciseDataset(Dataset[TrainingExample]):
    def __init__(
        self,
        samples: list[EncodedExercise],
        *,
        time_signature_vocabulary: TimeSignatureVocabulary,
        token_vocabulary: TokenVocabulary,
        musical_auxiliary_targets: MusicalAuxiliaryTargetConfig,
        conditioning: TrainingConditioningConfig,
        structural_control_vocabulary: StructuralControlVocabulary | None = None,
        include_structural_controls: bool = False,
        include_bar_count_control: bool = False,
        max_sequence_length: int | None = None,
    ) -> None:
        if include_structural_controls and structural_control_vocabulary is None:
            raise ValueError("structural_control_vocabulary is required when include_structural_controls is true")

        _log_skipped_sample_counts(
            samples,
            conditioning=conditioning,
            time_signature_vocabulary=time_signature_vocabulary,
            max_sequence_length=max_sequence_length,
        )
        harmonic_plan_context = _harmonic_plan_context(conditioning)

        self._examples = [
            _to_training_example(
                sample,
                conditioning=conditioning,
                include_structural_controls=include_structural_controls,
                include_bar_count_control=include_bar_count_control,
                time_signature_vocabulary=time_signature_vocabulary,
                token_vocabulary=token_vocabulary,
                musical_auxiliary_targets=musical_auxiliary_targets,
                structural_control_vocabulary=structural_control_vocabulary,
                harmonic_plan_context=harmonic_plan_context,
            )
            for sample in samples
            if _sample_is_usable(
                sample,
                conditioning=conditioning,
                time_signature_vocabulary=time_signature_vocabulary,
                max_sequence_length=max_sequence_length,
            )
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self._examples[index]


def _to_training_example(
    sample: EncodedExercise,
    *,
    conditioning: TrainingConditioningConfig,
    include_structural_controls: bool,
    include_bar_count_control: bool,
    time_signature_vocabulary: TimeSignatureVocabulary,
    token_vocabulary: TokenVocabulary,
    musical_auxiliary_targets: MusicalAuxiliaryTargetConfig,
    structural_control_vocabulary: StructuralControlVocabulary | None,
    harmonic_plan_context: _HarmonicPlanDatasetContext | None,
) -> TrainingExample:
    token_ids = torch.tensor(sample.token_ids, dtype=torch.long)
    bar_positions = torch.tensor(sample.bar_positions, dtype=torch.long)
    if token_ids.size(0) != bar_positions.size(0):
        raise ValueError(
            f"token_ids length {token_ids.size(0)} does not match bar_positions length {bar_positions.size(0)}"
        )

    input_token_ids = _prepend_start_token(token_ids, token_vocabulary=token_vocabulary)
    input_bar_positions = _prepend_start_bar_position(bar_positions)
    constraints = GenerationConstraints(
        time_numerator=sample.time_numerator,
        time_denominator=sample.time_denominator,
        bar_count=sample.metadata.bar_count,
        bar_durations=sample.metadata.bar_durations,
    )
    duration_vocabulary = token_vocabulary.duration_vocabulary
    tick_denominator = duration_tick_denominator(duration_vocabulary)
    input_coordinates = decoder_input_coordinates_from_token_ids(
        sample.token_ids[:-1],
        constraints=constraints,
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=tick_denominator,
    )
    structural_control_ids = _structural_control_ids(
        sample,
        include_structural_controls=include_structural_controls,
        include_bar_count_control=include_bar_count_control,
        structural_control_vocabulary=structural_control_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    difficulty_id = difficulty_level_to_id(sample.difficulty_level) if conditioning.use_difficulty else None
    scale_type_id = scale_type_to_id(sample.scale_type) if conditioning.use_scale_type else 0
    time_signature_id = (
        time_signature_to_id(
            (sample.time_numerator, sample.time_denominator),
            vocabulary=time_signature_vocabulary,
        )
        if conditioning.use_time_signature
        else 0
    )
    segment = sample.to_segment(token_vocabulary=token_vocabulary)
    harmonic_plan = _harmonic_plan_input_tensors(
        segment,
        context=harmonic_plan_context,
        constraints=constraints,
        coordinates=input_coordinates,
        tick_denominator=tick_denominator,
        token_vocabulary=token_vocabulary,
    )
    return TrainingExample(
        input_token_ids=input_token_ids,
        target_token_ids=token_ids,
        target_token_attributes=token_attribute_targets_from_token_ids(token_ids, vocabulary=token_vocabulary),
        musical_auxiliary_targets=musical_auxiliary_target_tensors_from_ids(
            musical_auxiliary_target_ids_from_difficulty_features(
                sample.metadata.difficulty_features,
                config=musical_auxiliary_targets,
            ),
            bar_target_ids=bar_musical_auxiliary_target_ids_from_segment(
                segment,
                duration_vocabulary=token_vocabulary.duration_vocabulary,
                config=musical_auxiliary_targets,
            ),
        ),
        target_bar_positions=bar_positions,
        bar_positions=input_bar_positions,
        bar_relative_ticks=torch.tensor(input_coordinates.bar_relative_ticks, dtype=torch.long),
        bar_duration_ticks=torch.tensor(input_coordinates.bar_duration_ticks, dtype=torch.long),
        active_hand_ids=torch.tensor(input_coordinates.active_hand_ids, dtype=torch.long),
        harmonic_plan=harmonic_plan,
        structural_control_ids=structural_control_ids,
        scale_root=sample.scale_root,
        scale_type_id=scale_type_to_id(sample.scale_type),
        time_numerator=sample.time_numerator,
        time_denominator=sample.time_denominator,
        bar_count=sample.metadata.bar_count,
        bar_durations=sample.metadata.bar_durations,
        difficulty_id=difficulty_id,
        conditioning_scale_type_id=scale_type_id,
        conditioning_time_signature_id=time_signature_id,
    )


def _harmonic_plan_context(conditioning: TrainingConditioningConfig) -> _HarmonicPlanDatasetContext | None:
    if not conditioning.use_harmony_conditioning:
        return None

    return _HarmonicPlanDatasetContext(
        decoder=ViterbiChordDecoder(ChordDecoderConfig.load()),
        vocabulary=ChordVocabularyConfig.load(),
    )


def _harmonic_plan_input_tensors(
    segment: Segment,
    *,
    context: _HarmonicPlanDatasetContext | None,
    constraints: GenerationConstraints,
    coordinates: DecoderInputCoordinates,
    tick_denominator: int,
    token_vocabulary: TokenVocabulary,
) -> HarmonicPlanInputTensors | None:
    if context is None:
        return None

    windows = harmonic_plan_windows_from_segment(
        segment,
        decoder=context.decoder,
        duration_vocabulary=token_vocabulary.duration_vocabulary,
        vocabulary=context.vocabulary,
    )
    return harmonic_plan_tensors_from_decoder_coordinates(
        windows,
        constraints=constraints,
        coordinates=coordinates,
        duration_tick_denominator=tick_denominator,
    )


def _log_skipped_sample_counts(
    samples: list[EncodedExercise],
    *,
    conditioning: TrainingConditioningConfig,
    time_signature_vocabulary: TimeSignatureVocabulary,
    max_sequence_length: int | None,
) -> None:
    unlabeled_difficulty_count = sum(
        1
        for sample in samples
        if conditioning.use_difficulty
        and sample.difficulty_level is None
        and _sample_has_allowed_length(sample, max_sequence_length=max_sequence_length)
    )
    if unlabeled_difficulty_count > 0:
        _LOGGER.warning(
            "Skipping %s training samples without difficulty labels because difficulty conditioning is enabled",
            unlabeled_difficulty_count,
        )

    overlength_count = sum(
        1 for sample in samples if max_sequence_length is not None and len(sample.token_ids) > max_sequence_length
    )
    if overlength_count > 0:
        _LOGGER.warning(
            "Skipping %s training samples longer than model max_sequence_length=%s",
            overlength_count,
            max_sequence_length,
        )

    unsupported_time_signature_count = sum(
        1
        for sample in samples
        if conditioning.use_time_signature
        and _sample_has_allowed_length(sample, max_sequence_length=max_sequence_length)
        and not time_signature_vocabulary.contains((sample.time_numerator, sample.time_denominator))
    )
    if unsupported_time_signature_count > 0:
        _LOGGER.warning(
            "Skipping %s training samples with time signatures outside the conditioning vocabulary",
            unsupported_time_signature_count,
        )


def _sample_is_usable(
    sample: EncodedExercise,
    *,
    conditioning: TrainingConditioningConfig,
    time_signature_vocabulary: TimeSignatureVocabulary,
    max_sequence_length: int | None,
) -> bool:
    return (
        _sample_has_allowed_length(sample, max_sequence_length=max_sequence_length)
        and (not conditioning.use_difficulty or sample.difficulty_level is not None)
        and (
            not conditioning.use_time_signature
            or time_signature_vocabulary.contains((sample.time_numerator, sample.time_denominator))
        )
    )


def _sample_has_allowed_length(sample: EncodedExercise, *, max_sequence_length: int | None) -> bool:
    return len(sample.token_ids) >= 1 and (max_sequence_length is None or len(sample.token_ids) <= max_sequence_length)


def _prepend_start_token(token_ids: Tensor, *, token_vocabulary: TokenVocabulary) -> Tensor:
    start_token = torch.tensor([token_vocabulary.start_token_id], dtype=token_ids.dtype)
    return torch.cat((start_token, token_ids[:-1]))


def _prepend_start_bar_position(bar_positions: Tensor) -> Tensor:
    start_position = torch.tensor([_START_BAR_POSITION], dtype=bar_positions.dtype)
    return torch.cat((start_position, bar_positions[:-1]))


def _structural_control_ids(
    sample: EncodedExercise,
    *,
    include_structural_controls: bool,
    include_bar_count_control: bool,
    structural_control_vocabulary: StructuralControlVocabulary | None,
    token_vocabulary: TokenVocabulary,
) -> Tensor:
    if not include_structural_controls:
        return torch.empty(0, dtype=torch.long)

    if structural_control_vocabulary is None:
        raise ValueError("structural_control_vocabulary is required")

    features = extract_structural_control_features(
        sample.to_segment(token_vocabulary=token_vocabulary),
        duration_vocabulary=token_vocabulary.duration_vocabulary,
    )
    if include_bar_count_control:
        features = features.model_copy(update={"bar_count": sample.metadata.bar_count})

    return torch.tensor(
        structural_control_vocabulary.features_to_ids(features),
        dtype=torch.long,
    )
