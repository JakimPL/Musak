from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from musak_model.training.conditioning import difficulty_level_to_id, scale_type_to_id, time_signature_to_id
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit

_PADDING_TOKEN_ID: int = 0
_PADDING_BAR_POSITION: int = -1

type _TrainingDataLoader = DataLoader[TrainingBatch]


@dataclass(frozen=True)
class TrainingExample:
    input_token_ids: Tensor
    target_token_ids: Tensor
    bar_positions: Tensor
    difficulty_id: int | None
    scale_type_id: int
    time_signature_id: int


@dataclass(frozen=True)
class TrainingBatch:
    input_token_ids: Tensor
    target_token_ids: Tensor
    bar_positions: Tensor
    token_padding_mask: Tensor
    difficulty_ids: Tensor | None
    scale_type_ids: Tensor
    time_signature_ids: Tensor


class EncodedExerciseDataset(Dataset[TrainingExample]):
    def __init__(self, samples: list[EncodedExercise], *, include_conditioning: bool = True) -> None:
        self._examples = [
            _to_training_example(sample, include_conditioning=include_conditioning)
            for sample in samples
            if len(sample.token_ids) >= 2
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
    include_conditioning: bool = True,
) -> tuple[DataLoader[TrainingBatch], DataLoader[TrainingBatch]]:
    train_dataset = EncodedExerciseDataset(split.train, include_conditioning=include_conditioning)
    validation_dataset = EncodedExerciseDataset(split.validation, include_conditioning=include_conditioning)
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

    for row_index, example in enumerate(examples):
        length = example.input_token_ids.size(0)
        input_token_ids[row_index, :length] = example.input_token_ids
        target_token_ids[row_index, :length] = example.target_token_ids
        bar_positions[row_index, :length] = example.bar_positions
        token_padding_mask[row_index, :length] = False

    difficulty_ids = _optional_tensor([example.difficulty_id for example in examples])
    scale_type_ids = torch.tensor([example.scale_type_id for example in examples], dtype=torch.long)
    time_signature_ids = torch.tensor([example.time_signature_id for example in examples], dtype=torch.long)

    return TrainingBatch(
        input_token_ids=input_token_ids,
        target_token_ids=target_token_ids,
        bar_positions=bar_positions,
        token_padding_mask=token_padding_mask,
        difficulty_ids=difficulty_ids,
        scale_type_ids=scale_type_ids,
        time_signature_ids=time_signature_ids,
    )


def _to_training_example(sample: EncodedExercise, *, include_conditioning: bool) -> TrainingExample:
    token_ids = torch.tensor(sample.token_ids, dtype=torch.long)
    bar_positions = torch.tensor(sample.bar_positions, dtype=torch.long)
    difficulty_id = difficulty_level_to_id(sample.difficulty_level) if include_conditioning else None
    scale_type_id = scale_type_to_id(sample.scale_type) if include_conditioning else 0
    time_signature_id = (
        time_signature_to_id((sample.time_numerator, sample.time_denominator)) if include_conditioning else 0
    )
    return TrainingExample(
        input_token_ids=token_ids[:-1],
        target_token_ids=token_ids[1:],
        bar_positions=bar_positions[:-1],
        difficulty_id=difficulty_id,
        scale_type_id=scale_type_id,
        time_signature_id=time_signature_id,
    )


def _optional_tensor(values: list[int | None]) -> Tensor | None:
    if any(value is None for value in values):
        return None

    return torch.tensor([value for value in values if value is not None], dtype=torch.long)
