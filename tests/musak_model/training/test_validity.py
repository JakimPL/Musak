from fractions import Fraction
from pathlib import Path

import torch

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.conditioning.time_signature import TimeSignatureVocabulary, TimeSignatureVocabularyConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, JoinWithPreviousToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.config import TrainingConditioningConfig
from musak_model.training.dataset.collate import collate_training_examples
from musak_model.training.dataset.examples import EncodedExerciseDataset
from musak_model.training.ingestion.schema import EncodedExercise
from musak_model.training.validity import TrainingValidityMaskBuilder


def _time_signature_vocabulary() -> TimeSignatureVocabulary:
    return TimeSignatureVocabulary(TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2))


def _conditioning_config() -> TrainingConditioningConfig:
    return TrainingConditioningConfig(
        use_time_signature=False,
        use_scale_type=False,
        use_difficulty=False,
        use_structural_conditioning=False,
        use_harmony_conditioning=False,
        use_validity_penalty=False,
        validity_penalty_weight=0.05,
    )


def _musical_auxiliary_target_config() -> MusicalAuxiliaryTargetConfig:
    return MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
        rhythmic_diversity_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        voice_independence_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        hand_span_bucket_boundaries=(3, 5, 8, 12, 16),
    )


def _sample(token_ids: list[int], bar_positions: list[int]) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=bar_positions,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
            difficulty_level=None,
        ),
    )


def _batch(token_ids: list[int], *, token_vocabulary: TokenVocabulary):
    dataset = EncodedExerciseDataset(
        [_sample(token_ids, [0] * len(token_ids))],
        time_signature_vocabulary=_time_signature_vocabulary(),
        token_vocabulary=token_vocabulary,
        musical_auxiliary_targets=_musical_auxiliary_target_config(),
        conditioning=_conditioning_config(),
    )
    return collate_training_examples([dataset[0]])


def _note_id(
    *,
    degree: int,
    duration_id: int,
    token_vocabulary: TokenVocabulary,
    octave_offset: int = 0,
) -> int:
    return token_vocabulary.token_to_id(
        NoteToken(degree=degree, accidental=0, octave_offset=octave_offset, duration_id=duration_id)
    )


def test_validity_mask_marks_early_bar_token_invalid(token_vocabulary: TokenVocabulary) -> None:
    batch = _batch([token_vocabulary.token_to_id(BarToken())], token_vocabulary=token_vocabulary)
    masks = TrainingValidityMaskBuilder(token_vocabulary).masks_for_batch(batch, device=torch.device("cpu"))

    assert masks.invalid_target_mask.tolist() == [[True]]


def test_validity_mask_marks_duplicate_chord_pitch_invalid(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    note_id = _note_id(degree=1, duration_id=whole_id, token_vocabulary=token_vocabulary)
    batch = _batch([note_id, note_id], token_vocabulary=token_vocabulary)

    masks = TrainingValidityMaskBuilder(token_vocabulary).masks_for_batch(batch, device=torch.device("cpu"))

    assert masks.invalid_token_mask[0, 1, note_id]


def test_validity_mask_marks_sixth_same_hand_chord_note_invalid(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    note_ids = [
        _note_id(degree=degree, duration_id=whole_id, token_vocabulary=token_vocabulary) for degree in range(1, 7)
    ]
    join_id = token_vocabulary.token_to_id(JoinWithPreviousToken())
    token_ids = [
        note_ids[0],
        note_ids[1],
        join_id,
        note_ids[2],
        join_id,
        note_ids[3],
        join_id,
        note_ids[4],
        join_id,
        note_ids[5],
    ]
    batch = _batch(token_ids, token_vocabulary=token_vocabulary)

    masks = TrainingValidityMaskBuilder(token_vocabulary).masks_for_batch(batch, device=torch.device("cpu"))

    assert masks.invalid_token_mask[0, 9, note_ids[5]]


def test_validity_mask_marks_onset_span_above_octave_invalid(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    root_id = _note_id(degree=1, duration_id=whole_id, token_vocabulary=token_vocabulary)
    ninth_id = _note_id(degree=2, duration_id=whole_id, token_vocabulary=token_vocabulary, octave_offset=1)
    batch = _batch([root_id, ninth_id], token_vocabulary=token_vocabulary)

    masks = TrainingValidityMaskBuilder(token_vocabulary).masks_for_batch(batch, device=torch.device("cpu"))

    assert masks.invalid_token_mask[0, 1, ninth_id]
