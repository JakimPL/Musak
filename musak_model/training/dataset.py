from dataclasses import dataclass
from typing import cast

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
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit

_PADDING_TOKEN_ID: int = 0
_PADDING_BAR_POSITION: int = -1
_START_BAR_POSITION: int = 0

type _TrainingDataLoader = DataLoader[TrainingBatch]


@dataclass(frozen=True)
class TrainingExample:
    input_token_ids: Tensor
    target_token_ids: Tensor
    bar_positions: Tensor
    structural_control_ids: Tensor
    difficulty_id: int | None
    scale_type_id: int
    time_signature_id: int


@dataclass(frozen=True)
class TrainingBatch:
    input_token_ids: Tensor
    target_token_ids: Tensor
    bar_positions: Tensor
    structural_control_ids: Tensor
    token_padding_mask: Tensor
    difficulty_ids: Tensor | None
    scale_type_ids: Tensor
    time_signature_ids: Tensor


class EncodedExerciseDataset(Dataset[TrainingExample]):
    def __init__(
        self,
        samples: list[EncodedExercise],
        *,
        time_signature_vocabulary: TimeSignatureVocabulary,
        token_vocabulary: TokenVocabulary,
        structural_control_vocabulary: StructuralControlVocabulary | None = None,
        include_conditioning: bool = True,
        include_structural_controls: bool = False,
    ) -> None:
        if include_structural_controls and structural_control_vocabulary is None:
            raise ValueError("structural_control_vocabulary is required when include_structural_controls is true")

        self._examples = [
            _to_training_example(
                sample,
                include_conditioning=include_conditioning,
                include_structural_controls=include_structural_controls,
                time_signature_vocabulary=time_signature_vocabulary,
                token_vocabulary=token_vocabulary,
                structural_control_vocabulary=structural_control_vocabulary,
            )
            for sample in samples
            if len(sample.token_ids) >= 1
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
    include_conditioning: bool = True,
    include_structural_controls: bool = False,
) -> tuple[DataLoader[TrainingBatch], DataLoader[TrainingBatch]]:
    train_dataset = EncodedExerciseDataset(
        split.train,
        include_conditioning=include_conditioning,
        include_structural_controls=include_structural_controls,
        time_signature_vocabulary=time_signature_vocabulary,
        token_vocabulary=token_vocabulary,
        structural_control_vocabulary=structural_control_vocabulary,
    )
    validation_dataset = EncodedExerciseDataset(
        split.validation,
        include_conditioning=include_conditioning,
        include_structural_controls=include_structural_controls,
        time_signature_vocabulary=time_signature_vocabulary,
        token_vocabulary=token_vocabulary,
        structural_control_vocabulary=structural_control_vocabulary,
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
    return cast(tuple[DataLoader[TrainingBatch], DataLoader[TrainingBatch]], (train_loader, validation_loader))


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
    scale_type_ids = torch.tensor([example.scale_type_id for example in examples], dtype=torch.long)
    time_signature_ids = torch.tensor([example.time_signature_id for example in examples], dtype=torch.long)

    return TrainingBatch(
        input_token_ids=input_token_ids,
        target_token_ids=target_token_ids,
        bar_positions=bar_positions,
        structural_control_ids=structural_control_ids,
        token_padding_mask=token_padding_mask,
        difficulty_ids=difficulty_ids,
        scale_type_ids=scale_type_ids,
        time_signature_ids=time_signature_ids,
    )


def _to_training_example(
    sample: EncodedExercise,
    *,
    include_conditioning: bool,
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
    difficulty_id = difficulty_level_to_id(sample.difficulty_level) if include_conditioning else None
    scale_type_id = scale_type_to_id(sample.scale_type) if include_conditioning else 0
    time_signature_id = (
        time_signature_to_id(
            (sample.time_numerator, sample.time_denominator),
            vocabulary=time_signature_vocabulary,
        )
        if include_conditioning
        else 0
    )
    return TrainingExample(
        input_token_ids=input_token_ids,
        target_token_ids=token_ids,
        bar_positions=input_bar_positions,
        structural_control_ids=structural_control_ids,
        difficulty_id=difficulty_id,
        scale_type_id=scale_type_id,
        time_signature_id=time_signature_id,
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
    return torch.tensor(structural_control_vocabulary.features_to_ids(features), dtype=torch.long)


def _optional_tensor(values: list[int | None]) -> Tensor | None:
    if any(value is None for value in values):
        return None

    return torch.tensor([value for value in values if value is not None], dtype=torch.long)
