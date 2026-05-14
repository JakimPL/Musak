from pathlib import Path

import pytest

from musak_model.data.schema import SegmentMetadata
from musak_model.tokens.schema import ScaleType
from musak_model.training.conditioning import difficulty_level_to_id, time_signature_to_id
from musak_model.training.dataset import EncodedExerciseDataset, collate_training_examples
from musak_model.training.ingestion.schema import EncodedExercise


def _sample(token_ids: list[int], bar_positions: list[int], *, difficulty_level: int | None = 3) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=bar_positions,
        metadata=SegmentMetadata(
            key_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
            difficulty_level=difficulty_level,
        ),
    )


def test_dataset_builds_teacher_forcing_examples() -> None:
    dataset = EncodedExerciseDataset([_sample([1, 2, 3], [0, 0, 0])])

    example = dataset[0]

    assert example.input_token_ids.tolist() == [1, 2]
    assert example.target_token_ids.tolist() == [2, 3]
    assert example.bar_positions.tolist() == [0, 0]


def test_dataset_skips_samples_too_short_for_next_token_training() -> None:
    dataset = EncodedExerciseDataset([_sample([1], [0])])

    assert len(dataset) == 0


def test_collate_pads_tokens_and_bar_positions() -> None:
    dataset = EncodedExerciseDataset(
        [
            _sample([1, 2, 3], [0, 0, 0]),
            _sample([4, 5], [0, 0]),
        ]
    )

    batch = collate_training_examples([dataset[0], dataset[1]])

    assert batch.input_token_ids.tolist() == [[1, 2], [4, 0]]
    assert batch.target_token_ids.tolist() == [[2, 3], [5, 0]]
    assert batch.bar_positions.tolist() == [[0, 0], [0, -1]]
    assert batch.token_padding_mask.tolist() == [[False, False], [False, True]]


def test_conditioning_ids_are_zero_based() -> None:
    assert difficulty_level_to_id(1) == 0
    assert difficulty_level_to_id(6) == 5
    assert difficulty_level_to_id(None) is None


def test_time_signature_mapping_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="unsupported time signature"):
        time_signature_to_id((5, 4))
