import logging
from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from musak_model.conditioning.structural import (
    StructuralControlVocabulary,
    extract_structural_control_features,
)
from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.conditioning import difficulty_level_to_id, scale_type_to_id, time_signature_to_id
from musak_model.training.config import TrainingConditioningConfig
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit

_PADDING_TOKEN_ID: Final[int] = 0
_PADDING_BAR_POSITION: Final[int] = -1
_START_BAR_POSITION: Final[int] = 0
_LOGGER = logging.getLogger(__name__)

type _TrainingDataLoader = DataLoader[TrainingBatch]


@dataclass(frozen=True)
class TrainingExample:
    input_token_ids: Tensor
    target_token_ids: Tensor
    bar_positions: Tensor
    structural_control_ids: Tensor
    key_root: int
    scale_type_id: int
    time_numerator: int
    time_denominator: int
    bar_count: int
    difficulty_id: int | None
    conditioning_scale_type_id: int
    conditioning_time_signature_id: int


@dataclass(frozen=True)
class TrainingBatch:
    input_token_ids: Tensor
    target_token_ids: Tensor
    bar_positions: Tensor
    structural_control_ids: Tensor
    key_roots: Tensor
    scale_type_ids: Tensor
    time_numerators: Tensor
    time_denominators: Tensor
    bar_counts: Tensor
    token_padding_mask: Tensor
    difficulty_ids: Tensor | None
    conditioning_scale_type_ids: Tensor
    conditioning_time_signature_ids: Tensor


class EncodedExerciseDataset(Dataset[TrainingExample]):
    def __init__(
        self,
        samples: list[EncodedExercise],
        *,
        time_signature_vocabulary: TimeSignatureVocabulary,
        token_vocabulary: TokenVocabulary,
        structural_control_vocabulary: StructuralControlVocabulary | None = None,
        conditioning: TrainingConditioningConfig = TrainingConditioningConfig(),
        include_structural_controls: bool = False,
        max_sequence_length: int | None = None,
    ) -> None:
        if include_structural_controls and structural_control_vocabulary is None:
            raise ValueError("structural_control_vocabulary is required when include_structural_controls is true")

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
            and len(sample.token_ids) >= 1
            and (max_sequence_length is None or len(sample.token_ids) <= max_sequence_length)
            and not time_signature_vocabulary.contains((sample.time_numerator, sample.time_denominator))
        )
        if unsupported_time_signature_count > 0:
            _LOGGER.warning(
                "Skipping %s training samples with time signatures outside the conditioning vocabulary",
                unsupported_time_signature_count,
            )

        self._examples = [
            _to_training_example(
                sample,
                conditioning=conditioning,
                include_structural_controls=include_structural_controls,
                time_signature_vocabulary=time_signature_vocabulary,
                token_vocabulary=token_vocabulary,
                structural_control_vocabulary=structural_control_vocabulary,
            )
            for sample in samples
            if len(sample.token_ids) >= 1
            and (max_sequence_length is None or len(sample.token_ids) <= max_sequence_length)
            and (
                not conditioning.use_time_signature
                or time_signature_vocabulary.contains((sample.time_numerator, sample.time_denominator))
            )
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self._examples[index]


def build_dataloaders(
    split: IngestionSplit,
    *,
    batch_size: int,
    shuffle_train: bool,
    num_workers: int,
    time_signature_vocabulary: TimeSignatureVocabulary,
    token_vocabulary: TokenVocabulary,
    structural_control_vocabulary: StructuralControlVocabulary | None = None,
    conditioning: TrainingConditioningConfig = TrainingConditioningConfig(),
    include_structural_controls: bool = False,
    max_sequence_length: int | None = None,
) -> tuple[DataLoader[TrainingBatch], DataLoader[TrainingBatch]]:
    train_dataset = EncodedExerciseDataset(
        split.train,
        conditioning=conditioning,
        include_structural_controls=include_structural_controls,
        time_signature_vocabulary=time_signature_vocabulary,
        token_vocabulary=token_vocabulary,
        structural_control_vocabulary=structural_control_vocabulary,
        max_sequence_length=max_sequence_length,
    )
    validation_dataset = EncodedExerciseDataset(
        split.validation,
        conditioning=conditioning,
        include_structural_controls=include_structural_controls,
        time_signature_vocabulary=time_signature_vocabulary,
        token_vocabulary=token_vocabulary,
        structural_control_vocabulary=structural_control_vocabulary,
        max_sequence_length=max_sequence_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_training_examples,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_training_examples,
    )
    return cast(
        tuple[DataLoader[TrainingBatch], DataLoader[TrainingBatch]],
        (
            train_loader,
            validation_loader,
        ),
    )


def collate_training_examples(examples: list[TrainingExample]) -> TrainingBatch:
    if not examples:
        raise ValueError("cannot collate empty batch")

    max_length = max(example.input_token_ids.size(0) for example in examples)
    input_token_ids = torch.full((len(examples), max_length), _PADDING_TOKEN_ID, dtype=torch.long)
    target_token_ids = torch.full((len(examples), max_length), _PADDING_TOKEN_ID, dtype=torch.long)
    bar_positions = torch.full((len(examples), max_length), _PADDING_BAR_POSITION, dtype=torch.long)
    token_padding_mask = torch.ones((len(examples), max_length), dtype=torch.bool)
    structural_control_count = examples[0].structural_control_ids.size(0)
    structural_control_ids = torch.zeros((len(examples), structural_control_count), dtype=torch.long)
    key_roots = torch.tensor([example.key_root for example in examples], dtype=torch.long)
    scale_type_ids = torch.tensor([example.scale_type_id for example in examples], dtype=torch.long)
    time_numerators = torch.tensor([example.time_numerator for example in examples], dtype=torch.long)
    time_denominators = torch.tensor([example.time_denominator for example in examples], dtype=torch.long)
    bar_counts = torch.tensor([example.bar_count for example in examples], dtype=torch.long)

    for row_index, example in enumerate(examples):
        length = example.input_token_ids.size(0)
        input_token_ids[row_index, :length] = example.input_token_ids
        target_token_ids[row_index, :length] = example.target_token_ids
        bar_positions[row_index, :length] = example.bar_positions
        token_padding_mask[row_index, :length] = False
        if example.structural_control_ids.size(0) != structural_control_count:
            raise ValueError("all examples must have the same number of structural controls")

        structural_control_ids[row_index] = example.structural_control_ids

    difficulty_ids = _optional_tensor([example.difficulty_id for example in examples])
    conditioning_scale_type_ids = torch.tensor(
        [example.conditioning_scale_type_id for example in examples],
        dtype=torch.long,
    )
    conditioning_time_signature_ids = torch.tensor(
        [example.conditioning_time_signature_id for example in examples],
        dtype=torch.long,
    )

    return TrainingBatch(
        input_token_ids=input_token_ids,
        target_token_ids=target_token_ids,
        bar_positions=bar_positions,
        structural_control_ids=structural_control_ids,
        key_roots=key_roots,
        scale_type_ids=scale_type_ids,
        time_numerators=time_numerators,
        time_denominators=time_denominators,
        bar_counts=bar_counts,
        token_padding_mask=token_padding_mask,
        difficulty_ids=difficulty_ids,
        conditioning_scale_type_ids=conditioning_scale_type_ids,
        conditioning_time_signature_ids=conditioning_time_signature_ids,
    )


def _to_training_example(
    sample: EncodedExercise,
    *,
    conditioning: TrainingConditioningConfig,
    include_structural_controls: bool,
    time_signature_vocabulary: TimeSignatureVocabulary,
    token_vocabulary: TokenVocabulary,
    structural_control_vocabulary: StructuralControlVocabulary | None,
) -> TrainingExample:
    token_ids = torch.tensor(sample.token_ids, dtype=torch.long)
    bar_positions = torch.tensor(sample.bar_positions, dtype=torch.long)
    if token_ids.size(0) != bar_positions.size(0):
        raise ValueError(
            f"token_ids length {token_ids.size(0)} does not match bar_positions length {bar_positions.size(0)}"
        )

    input_token_ids = _prepend_start_token(token_ids, token_vocabulary=token_vocabulary)
    input_bar_positions = _prepend_start_bar_position(bar_positions)
    structural_control_ids = _structural_control_ids(
        sample,
        include_structural_controls=include_structural_controls,
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
    return TrainingExample(
        input_token_ids=input_token_ids,
        target_token_ids=token_ids,
        bar_positions=input_bar_positions,
        structural_control_ids=structural_control_ids,
        key_root=sample.key_root,
        scale_type_id=scale_type_to_id(sample.scale_type),
        time_numerator=sample.time_numerator,
        time_denominator=sample.time_denominator,
        bar_count=sample.metadata.bar_count,
        difficulty_id=difficulty_id,
        conditioning_scale_type_id=scale_type_id,
        conditioning_time_signature_id=time_signature_id,
    )


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

    return torch.tensor(
        structural_control_vocabulary.features_to_ids(features),
        dtype=torch.long,
    )


def _optional_tensor(values: list[int | None]) -> Tensor | None:
    if any(value is None for value in values):
        return None

    return torch.tensor(
        [value for value in values if value is not None],
        dtype=torch.long,
    )
