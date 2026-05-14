from pathlib import Path

import pytest

from musak_model.conditioning.time_signature import TimeSignatureVocabulary, TimeSignatureVocabularyConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.tokens.schema import ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.conditioning import difficulty_level_to_id, time_signature_to_id
from musak_model.training.dataset import EncodedExerciseDataset, collate_training_examples
from musak_model.training.ingestion.schema import EncodedExercise


def _time_signature_vocabulary() -> TimeSignatureVocabulary:
    return TimeSignatureVocabulary(TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2))


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


def test_dataset_builds_teacher_forcing_examples_with_start_token(token_vocabulary: TokenVocabulary) -> None:
    dataset = EncodedExerciseDataset(
        [_sample([1, 2, 3], [0, 0, 0])],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
    )

    example = dataset[0]

    assert example.input_token_ids.tolist() == [token_vocabulary.start_token_id, 1, 2]
    assert example.target_token_ids.tolist() == [1, 2, 3]
    assert example.bar_positions.tolist() == [0, 0, 0]


def test_dataset_keeps_single_token_samples_for_start_token_training(token_vocabulary: TokenVocabulary) -> None:
    dataset = EncodedExerciseDataset(
        [_sample([1], [0])],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
    )

    assert len(dataset) == 1
    assert dataset[0].input_token_ids.tolist() == [token_vocabulary.start_token_id]
    assert dataset[0].target_token_ids.tolist() == [1]


def test_dataset_skips_empty_samples(token_vocabulary: TokenVocabulary) -> None:
    dataset = EncodedExerciseDataset(
        [_sample([], [])],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
    )

    assert len(dataset) == 0


def test_dataset_rejects_mismatched_token_and_bar_position_lengths(token_vocabulary: TokenVocabulary) -> None:
    with pytest.raises(ValueError, match="bar_positions"):
        EncodedExerciseDataset(
            [_sample([1, 2], [0])],
            time_signature_vocabulary=_time_signature_vocabulary(),
            token_vocabulary=token_vocabulary,
        )


def test_collate_pads_tokens_and_bar_positions(token_vocabulary: TokenVocabulary) -> None:
    dataset = EncodedExerciseDataset(
        [
            _sample([1, 2, 3], [0, 0, 0]),
            _sample([4, 5], [0, 0]),
        ],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
    )

    batch = collate_training_examples([dataset[0], dataset[1]])

    assert batch.input_token_ids.tolist() == [
        [token_vocabulary.start_token_id, 1, 2],
        [token_vocabulary.start_token_id, 4, 0],
    ]
    assert batch.target_token_ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch.bar_positions.tolist() == [[0, 0, 0], [0, 0, -1]]
    assert batch.token_padding_mask.tolist() == [[False, False, False], [False, False, True]]


def test_conditioning_ids_are_zero_based() -> None:
    assert difficulty_level_to_id(0) == 0
    assert difficulty_level_to_id(5) == 5
    assert difficulty_level_to_id(None) is None


def test_time_signature_mapping_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="unsupported time signature"):
        time_signature_to_id((8, 4), vocabulary=_time_signature_vocabulary())
