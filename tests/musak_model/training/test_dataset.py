from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.conditioning.config import ConditioningConfig
from musak_model.conditioning.structural import UNKNOWN_CONTROL_ID, StructuralControlName, StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary, TimeSignatureVocabularyConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.conditioning import difficulty_level_to_id, scale_type_to_id, time_signature_to_id
from musak_model.training.config import TrainingConditioningConfig
from musak_model.training.dataset import EncodedExerciseDataset, collate_training_examples
from musak_model.training.ingestion.schema import EncodedExercise


def _time_signature_vocabulary() -> TimeSignatureVocabulary:
    return TimeSignatureVocabulary(TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2))


def _sample(
    token_ids: list[int],
    bar_positions: list[int],
    *,
    difficulty_level: int | None = 3,
    time_signature: tuple[int, int] = (4, 4),
    bar_count: int = 1,
) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=bar_positions,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=time_signature[0],
            time_denominator=time_signature[1],
            bar_count=bar_count,
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


def test_dataset_skips_samples_longer_than_max_sequence_length(token_vocabulary: TokenVocabulary) -> None:
    dataset = EncodedExerciseDataset(
        [
            _sample([1, 2, 3], [0, 0, 0]),
            _sample([1, 2, 3, 4], [0, 0, 0, 0]),
        ],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
        max_sequence_length=3,
    )

    assert len(dataset) == 1
    assert dataset[0].target_token_ids.tolist() == [1, 2, 3]


def test_dataset_skips_samples_with_unsupported_time_signature_when_conditioned(
    token_vocabulary: TokenVocabulary,
) -> None:
    dataset = EncodedExerciseDataset(
        [
            _sample([1, 2, 3], [0, 0, 0], time_signature=(4, 4)),
            _sample([4, 5, 6], [0, 0, 0], time_signature=(2, 1)),
        ],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
        conditioning=TrainingConditioningConfig(use_time_signature=True),
    )

    assert len(dataset) == 1
    assert dataset[0].target_token_ids.tolist() == [1, 2, 3]


def test_dataset_keeps_unsupported_time_signature_when_time_signature_conditioning_disabled(
    token_vocabulary: TokenVocabulary,
) -> None:
    dataset = EncodedExerciseDataset(
        [_sample([1, 2, 3], [0, 0, 0], time_signature=(2, 1))],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
        conditioning=TrainingConditioningConfig(use_time_signature=False, use_scale_type=True),
    )

    assert len(dataset) == 1
    assert dataset[0].conditioning_time_signature_id == 0
    assert dataset[0].scale_type_id == scale_type_to_id(ScaleType.MAJOR)


def test_dataset_builds_independent_metadata_conditioning_ids(token_vocabulary: TokenVocabulary) -> None:
    dataset = EncodedExerciseDataset(
        [_sample([1, 2, 3], [0, 0, 0], difficulty_level=3)],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
        conditioning=TrainingConditioningConfig(
            use_time_signature=True,
            use_scale_type=True,
            use_difficulty=True,
        ),
    )

    example = dataset[0]

    assert example.difficulty_id == 3
    assert example.scale_type_id == scale_type_to_id(ScaleType.MAJOR)
    assert example.conditioning_time_signature_id == time_signature_to_id(
        (4, 4),
        vocabulary=_time_signature_vocabulary(),
    )


def test_dataset_omits_difficulty_when_disabled(token_vocabulary: TokenVocabulary) -> None:
    dataset = EncodedExerciseDataset(
        [_sample([1, 2, 3], [0, 0, 0], difficulty_level=3)],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
        conditioning=TrainingConditioningConfig(use_time_signature=True, use_scale_type=True),
    )

    batch = collate_training_examples([dataset[0]])

    assert dataset[0].difficulty_id is None
    assert batch.difficulty_ids is None


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
    assert batch.structural_control_ids.tolist() == [[], []]
    assert batch.token_padding_mask.tolist() == [[False, False, False], [False, False, True]]


def test_dataset_builds_structural_control_ids_when_enabled(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    token_ids = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        ]
    )
    dataset = EncodedExerciseDataset(
        [_sample(token_ids, [0, 0])],
        include_structural_controls=True,
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
        structural_control_vocabulary=StructuralControlVocabulary(ConditioningConfig.load().structural),
    )

    assert dataset[0].structural_control_ids.numel() == 8
    assert dataset[0].structural_control_ids.tolist()[1] == 1
    assert dataset[0].structural_control_ids.tolist()[-1] == UNKNOWN_CONTROL_ID


def test_dataset_uses_bar_count_control_only_when_enabled(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    token_ids = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        ]
    )
    structural_control_vocabulary = StructuralControlVocabulary(ConditioningConfig.load().structural)
    dataset = EncodedExerciseDataset(
        [_sample(token_ids, [0, 0], bar_count=4)],
        include_structural_controls=True,
        include_bar_count_control=True,
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
        structural_control_vocabulary=structural_control_vocabulary,
    )
    bar_count_index = structural_control_vocabulary.control_index(StructuralControlName.BAR_COUNT)

    assert dataset[0].structural_control_ids.tolist()[bar_count_index] == 3


def test_dataset_requires_structural_vocabulary_when_structural_controls_are_enabled(
    token_vocabulary: TokenVocabulary,
) -> None:
    with pytest.raises(ValueError, match="structural_control_vocabulary"):
        EncodedExerciseDataset(
            [_sample([1], [0])],
            include_structural_controls=True,
            time_signature_vocabulary=_time_signature_vocabulary(),
            token_vocabulary=token_vocabulary,
        )


def test_conditioning_ids_are_zero_based() -> None:
    assert difficulty_level_to_id(0) == 0
    assert difficulty_level_to_id(5) == 5
    assert difficulty_level_to_id(None) is None


def test_time_signature_mapping_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="unsupported time signature"):
        time_signature_to_id((8, 4), vocabulary=_time_signature_vocabulary())
