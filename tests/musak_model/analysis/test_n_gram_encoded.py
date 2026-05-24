from fractions import Fraction
from pathlib import Path

from musak_model.analysis.n_grams import (
    count_encoded_exercise_figure_ngrams,
    count_encoded_exercises_figure_ngrams,
)
from musak_model.data.schema import SegmentMetadata
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def test_count_encoded_exercise_figure_ngrams_decodes_tokens(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    sample = _sample(
        token_vocabulary.encode(
            [
                HandToken(hand=Hand.RIGHT),
                _note(1, duration_id=quarter_id),
                _note(2, duration_id=quarter_id),
            ]
        ),
        scale_type=ScaleType.MAJOR,
    )

    counts = count_encoded_exercise_figure_ngrams(
        sample,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=2,
        max_n=2,
    )

    assert sum(counts[Hand.RIGHT][2].values()) == 1
    assert sum(counts[Hand.LEFT][2].values()) == 0


def test_count_encoded_exercises_figure_ngrams_keeps_scale_types_separate(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    tokens = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, duration_id=quarter_id),
            _note(2, duration_id=quarter_id),
        ]
    )

    counts = count_encoded_exercises_figure_ngrams(
        [
            _sample(tokens, scale_type=ScaleType.MAJOR),
            _sample(tokens, scale_type=ScaleType.HARMONIC_MINOR),
        ],
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=2,
        max_n=2,
    )

    assert set(counts) == {ScaleType.MAJOR, ScaleType.HARMONIC_MINOR}
    assert sum(counts[ScaleType.MAJOR][Hand.RIGHT][2].values()) == 1
    assert sum(counts[ScaleType.HARMONIC_MINOR][Hand.RIGHT][2].values()) == 1


def test_count_encoded_exercises_figure_ngrams_merges_same_scale_counts(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    tokens = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, duration_id=quarter_id),
            _note(2, duration_id=quarter_id),
        ]
    )

    counts = count_encoded_exercises_figure_ngrams(
        [
            _sample(tokens, scale_type=ScaleType.MAJOR),
            _sample(tokens, scale_type=ScaleType.MAJOR),
        ],
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=2,
        max_n=2,
    )

    assert sum(counts[ScaleType.MAJOR][Hand.RIGHT][2].values()) == 2


def _sample(token_ids: list[int], *, scale_type: ScaleType) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=[0 for _ in token_ids],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=scale_type,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("sample.mxl"),
        ),
    )


def _note(degree: int, *, duration_id: int) -> NoteToken:
    return NoteToken(
        degree=degree,
        accidental=0,
        octave_offset=0,
        duration_id=duration_id,
    )
