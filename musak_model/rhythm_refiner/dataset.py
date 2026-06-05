from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from musak_model.rhythm_refiner.config import (
    RhythmRefinerDataConfig,
    RhythmRefinerMaskingConfig,
    RhythmRefinerModelConfig,
)
from musak_model.rhythm_refiner.extraction import rhythm_grid_from_segment
from musak_model.rhythm_refiner.schema import RhythmGridConfig, RhythmGridFrame
from musak_model.rhythm_refiner.vocabulary import (
    COACTIVITY_TARGET_IGNORE_ID,
    RHYTHM_INPUT_UNKNOWN_ID,
    RHYTHM_TARGET_IGNORE_ID,
    coactivity_target_state_id,
    rhythm_input_state_id,
    rhythm_target_state_id,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise
from musak_model.training.progress import progress

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RhythmRefinerExample:
    input_state_ids: Tensor
    target_state_ids: Tensor
    coactivity_target_ids: Tensor
    cell_index_ids: Tensor
    bar_index_ids: Tensor
    distance_to_end_ids: Tensor
    bar_duration_cell_count_ids: Tensor
    time_numerator_id: int
    time_denominator_id: int
    activity_loss_mask: Tensor
    coactivity_loss_mask: Tensor


@dataclass(frozen=True)
class RhythmRefinerBatch:
    input_state_ids: Tensor
    target_state_ids: Tensor
    coactivity_target_ids: Tensor
    cell_index_ids: Tensor
    bar_index_ids: Tensor
    distance_to_end_ids: Tensor
    bar_duration_cell_count_ids: Tensor
    time_numerator_ids: Tensor
    time_denominator_ids: Tensor
    activity_loss_mask: Tensor
    coactivity_loss_mask: Tensor
    padding_mask: Tensor

    def to(self, device: torch.device) -> RhythmRefinerBatch:
        return RhythmRefinerBatch(
            input_state_ids=self.input_state_ids.to(device),
            target_state_ids=self.target_state_ids.to(device),
            coactivity_target_ids=self.coactivity_target_ids.to(device),
            cell_index_ids=self.cell_index_ids.to(device),
            bar_index_ids=self.bar_index_ids.to(device),
            distance_to_end_ids=self.distance_to_end_ids.to(device),
            bar_duration_cell_count_ids=self.bar_duration_cell_count_ids.to(device),
            time_numerator_ids=self.time_numerator_ids.to(device),
            time_denominator_ids=self.time_denominator_ids.to(device),
            activity_loss_mask=self.activity_loss_mask.to(device),
            coactivity_loss_mask=self.coactivity_loss_mask.to(device),
            padding_mask=self.padding_mask.to(device),
        )


class RhythmRefinerDataset(Dataset[RhythmRefinerExample]):
    def __init__(
        self,
        frames: tuple[RhythmGridFrame, ...],
        *,
        masking: RhythmRefinerMaskingConfig,
        model_config: RhythmRefinerModelConfig,
    ) -> None:
        self._frames = frames
        self._masking = masking
        self._model_config = model_config

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> RhythmRefinerExample:
        frame = self._frames[index]
        target_state_ids = _target_state_ids(frame)
        activity_loss_mask = _activity_loss_mask(target_state_ids, masking=self._masking, index=index)
        input_state_ids = _input_state_ids(frame, activity_loss_mask=activity_loss_mask)
        return RhythmRefinerExample(
            input_state_ids=input_state_ids,
            target_state_ids=target_state_ids,
            coactivity_target_ids=torch.tensor(
                [coactivity_target_state_id(state) for state in frame.coactivity_states],
                dtype=torch.long,
            ),
            cell_index_ids=torch.tensor(
                [_capped_id(cell.cell_index, self._model_config.max_cells_per_bar) for cell in frame.cells],
                dtype=torch.long,
            ),
            bar_index_ids=torch.tensor(
                [_capped_id(cell.bar_index, self._model_config.max_bar_count) for cell in frame.cells],
                dtype=torch.long,
            ),
            distance_to_end_ids=torch.tensor(
                [_capped_id(cell.distance_to_end, self._model_config.max_distance_cells) for cell in frame.cells],
                dtype=torch.long,
            ),
            bar_duration_cell_count_ids=torch.tensor(
                [
                    _capped_id(_bar_duration_cell_count(frame, cell.bar_index), self._model_config.max_cells_per_bar)
                    for cell in frame.cells
                ],
                dtype=torch.long,
            ),
            time_numerator_id=_capped_id(frame.time_numerator, self._model_config.max_time_numerator),
            time_denominator_id=_capped_id(frame.time_denominator, self._model_config.max_time_denominator),
            activity_loss_mask=activity_loss_mask,
            coactivity_loss_mask=activity_loss_mask.any(dim=-1),
        )


def rhythm_refiner_frames_from_samples(
    samples: list[EncodedExercise],
    *,
    token_vocabulary: TokenVocabulary,
    duration_vocabulary: DurationVocabulary,
    grid_config: RhythmGridConfig,
    data_config: RhythmRefinerDataConfig,
    show_progress: bool,
) -> tuple[RhythmGridFrame, ...]:
    frames: list[RhythmGridFrame] = []
    skipped_count = 0
    for sample in progress(
        samples,
        description="Building rhythm refiner frames",
        unit="sample",
        enabled=show_progress,
        total=len(samples),
    ):
        try:
            frame = rhythm_grid_from_segment(
                sample.to_segment(token_vocabulary=token_vocabulary),
                duration_vocabulary=duration_vocabulary,
                config=grid_config,
            )
        except ValueError as exception:
            skipped_count += 1
            _LOGGER.debug("Skipping rhythm refiner sample %s: %s", sample.source_file, exception)
            continue

        if data_config.max_cells is not None and len(frame.cells) > data_config.max_cells:
            skipped_count += 1
            continue
        if frame.cells:
            frames.append(frame)

    if not frames:
        raise ValueError("no rhythm refiner frames were built from encoded samples")

    if skipped_count:
        _LOGGER.warning(
            "Skipped %s rhythm refiner sample(s) that were incompatible with grid or data limits",
            skipped_count,
        )

    return tuple(frames)


def collate_rhythm_refiner_examples(examples: list[RhythmRefinerExample]) -> RhythmRefinerBatch:
    if not examples:
        raise ValueError("cannot collate empty rhythm refiner batch")

    batch_size = len(examples)
    max_length = max(example.input_state_ids.size(0) for example in examples)
    input_state_ids = torch.full((batch_size, max_length, 2), RHYTHM_INPUT_UNKNOWN_ID, dtype=torch.long)
    target_state_ids = torch.full((batch_size, max_length, 2), RHYTHM_TARGET_IGNORE_ID, dtype=torch.long)
    coactivity_target_ids = torch.full((batch_size, max_length), COACTIVITY_TARGET_IGNORE_ID, dtype=torch.long)
    cell_index_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    bar_index_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    distance_to_end_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    bar_duration_cell_count_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    activity_loss_mask = torch.zeros((batch_size, max_length, 2), dtype=torch.bool)
    coactivity_loss_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)

    for row_index, example in enumerate(examples):
        length = example.input_state_ids.size(0)
        input_state_ids[row_index, :length] = example.input_state_ids
        target_state_ids[row_index, :length] = example.target_state_ids
        coactivity_target_ids[row_index, :length] = example.coactivity_target_ids
        cell_index_ids[row_index, :length] = example.cell_index_ids
        bar_index_ids[row_index, :length] = example.bar_index_ids
        distance_to_end_ids[row_index, :length] = example.distance_to_end_ids
        bar_duration_cell_count_ids[row_index, :length] = example.bar_duration_cell_count_ids
        activity_loss_mask[row_index, :length] = example.activity_loss_mask
        coactivity_loss_mask[row_index, :length] = example.coactivity_loss_mask
        padding_mask[row_index, :length] = False

    return RhythmRefinerBatch(
        input_state_ids=input_state_ids,
        target_state_ids=target_state_ids,
        coactivity_target_ids=coactivity_target_ids,
        cell_index_ids=cell_index_ids,
        bar_index_ids=bar_index_ids,
        distance_to_end_ids=distance_to_end_ids,
        bar_duration_cell_count_ids=bar_duration_cell_count_ids,
        time_numerator_ids=torch.tensor([example.time_numerator_id for example in examples], dtype=torch.long),
        time_denominator_ids=torch.tensor([example.time_denominator_id for example in examples], dtype=torch.long),
        activity_loss_mask=activity_loss_mask,
        coactivity_loss_mask=coactivity_loss_mask,
        padding_mask=padding_mask,
    )


def _target_state_ids(frame: RhythmGridFrame) -> Tensor:
    return torch.tensor(
        [
            [rhythm_target_state_id(right_state), rhythm_target_state_id(left_state)]
            for right_state, left_state in zip(frame.right_hand_states, frame.left_hand_states, strict=True)
        ],
        dtype=torch.long,
    )


def _input_state_ids(frame: RhythmGridFrame, *, activity_loss_mask: Tensor) -> Tensor:
    input_ids = torch.tensor(
        [
            [rhythm_input_state_id(right_state), rhythm_input_state_id(left_state)]
            for right_state, left_state in zip(frame.right_hand_states, frame.left_hand_states, strict=True)
        ],
        dtype=torch.long,
    )
    return input_ids.masked_fill(activity_loss_mask, RHYTHM_INPUT_UNKNOWN_ID)


def _activity_loss_mask(
    target_state_ids: Tensor,
    *,
    masking: RhythmRefinerMaskingConfig,
    index: int,
) -> Tensor:
    generator = torch.Generator()
    generator.manual_seed(masking.seed + index)
    mask = torch.rand(target_state_ids.shape, generator=generator) < masking.mask_probability
    if not bool(mask.any()):
        flat_index = int(torch.randint(target_state_ids.numel(), (1,), generator=generator).item())
        mask.view(-1)[flat_index] = True
    return mask


def _bar_duration_cell_count(frame: RhythmGridFrame, bar_index: int) -> int:
    cell_count = frame.bar_durations[bar_index] * frame.config.grid_denominator
    if cell_count.denominator != 1:
        raise ValueError("bar duration cannot be represented as rhythm-grid cells")
    return cell_count.numerator


def _capped_id(value: int, maximum: int) -> int:
    return min(value, maximum)
