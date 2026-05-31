from typing import Final

import torch
from torch import Tensor

from musak_model.auxiliary.targets import stack_musical_auxiliary_targets
from musak_model.training.dataset.factorized import pad_token_attribute_targets
from musak_model.training.dataset.schema import TrainingBatch, TrainingExample

_PADDING_TOKEN_ID: Final[int] = 0
_PADDING_BAR_POSITION: Final[int] = -1
_PADDING_BAR_RELATIVE_TICKS: Final[int] = -1
_PADDING_BAR_DURATION_TICKS: Final[int] = 1
_PADDING_ACTIVE_HAND_ID: Final[int] = -1


def collate_training_examples(examples: list[TrainingExample]) -> TrainingBatch:
    if not examples:
        raise ValueError("cannot collate empty batch")

    max_length = max(example.input_token_ids.size(0) for example in examples)
    input_token_ids = torch.full((len(examples), max_length), _PADDING_TOKEN_ID, dtype=torch.long)
    target_token_ids = torch.full((len(examples), max_length), _PADDING_TOKEN_ID, dtype=torch.long)
    bar_positions = torch.full((len(examples), max_length), _PADDING_BAR_POSITION, dtype=torch.long)
    bar_relative_ticks = torch.full((len(examples), max_length), _PADDING_BAR_RELATIVE_TICKS, dtype=torch.long)
    bar_duration_ticks = torch.full((len(examples), max_length), _PADDING_BAR_DURATION_TICKS, dtype=torch.long)
    active_hand_ids = torch.full((len(examples), max_length), _PADDING_ACTIVE_HAND_ID, dtype=torch.long)
    token_padding_mask = torch.ones((len(examples), max_length), dtype=torch.bool)
    structural_control_count = examples[0].structural_control_ids.size(0)
    structural_control_ids = torch.zeros((len(examples), structural_control_count), dtype=torch.long)
    scale_roots = torch.tensor([example.scale_root for example in examples], dtype=torch.long)
    scale_type_ids = torch.tensor([example.scale_type_id for example in examples], dtype=torch.long)
    time_numerators = torch.tensor([example.time_numerator for example in examples], dtype=torch.long)
    time_denominators = torch.tensor([example.time_denominator for example in examples], dtype=torch.long)
    bar_counts = torch.tensor([example.bar_count for example in examples], dtype=torch.long)
    bar_durations = tuple(example.bar_durations for example in examples)

    for row_index, example in enumerate(examples):
        length = example.input_token_ids.size(0)
        input_token_ids[row_index, :length] = example.input_token_ids
        target_token_ids[row_index, :length] = example.target_token_ids
        bar_positions[row_index, :length] = example.bar_positions
        bar_relative_ticks[row_index, :length] = example.bar_relative_ticks
        bar_duration_ticks[row_index, :length] = example.bar_duration_ticks
        active_hand_ids[row_index, :length] = example.active_hand_ids
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
        target_token_attributes=pad_token_attribute_targets(
            [example.target_token_attributes for example in examples],
            max_length=max_length,
        ),
        musical_auxiliary_targets=stack_musical_auxiliary_targets(
            [example.musical_auxiliary_targets for example in examples]
        ),
        bar_positions=bar_positions,
        bar_relative_ticks=bar_relative_ticks,
        bar_duration_ticks=bar_duration_ticks,
        active_hand_ids=active_hand_ids,
        structural_control_ids=structural_control_ids,
        scale_roots=scale_roots,
        scale_type_ids=scale_type_ids,
        time_numerators=time_numerators,
        time_denominators=time_denominators,
        bar_counts=bar_counts,
        bar_durations=bar_durations,
        token_padding_mask=token_padding_mask,
        difficulty_ids=difficulty_ids,
        conditioning_scale_type_ids=conditioning_scale_type_ids,
        conditioning_time_signature_ids=conditioning_time_signature_ids,
    )


def _optional_tensor(values: list[int | None]) -> Tensor | None:
    if any(value is None for value in values):
        return None

    return torch.tensor(
        [value for value in values if value is not None],
        dtype=torch.long,
    )
