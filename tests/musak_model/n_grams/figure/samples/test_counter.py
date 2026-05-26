from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import SegmentMetadata
from musak_model.n_grams.figure.samples.counter import (
    count_encoded_exercises_figure_n_grams,
    count_encoded_exercises_figure_ngrams_with_samples,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def test_count_encoded_exercises_figure_ngrams_keeps_scale_types_separate(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    tokens = _encoded_tokens(token_vocabulary, quarter_id=quarter_id)

    counts = count_encoded_exercises_figure_n_grams(
        [
            _sample(tokens, scale_type=ScaleType.MAJOR),
            _sample(tokens, scale_type=ScaleType.HARMONIC_MINOR),
        ],
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=2,
        max_n=2,
        workers=1,
        batch_size=1,
    )

    assert set(counts) == {ScaleType.MAJOR, ScaleType.HARMONIC_MINOR}
    assert sum(counts[ScaleType.MAJOR][Hand.RIGHT][2].values()) == 1
    assert sum(counts[ScaleType.HARMONIC_MINOR][Hand.RIGHT][2].values()) == 1


def test_count_encoded_exercises_figure_ngrams_merges_same_scale_counts(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    tokens = _encoded_tokens(token_vocabulary, quarter_id=quarter_id)

    counts = count_encoded_exercises_figure_n_grams(
        [
            _sample(tokens, scale_type=ScaleType.MAJOR),
            _sample(tokens, scale_type=ScaleType.MAJOR),
        ],
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=2,
        max_n=2,
        workers=1,
        batch_size=1,
    )

    assert sum(counts[ScaleType.MAJOR][Hand.RIGHT][2].values()) == 2


def test_count_encoded_exercises_figure_ngrams_parallel_matches_serial(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    tokens = _encoded_tokens(token_vocabulary, quarter_id=quarter_id)
    samples = [
        _sample(tokens, scale_type=ScaleType.MAJOR),
        _sample(tokens, scale_type=ScaleType.MAJOR),
        _sample(tokens, scale_type=ScaleType.HARMONIC_MINOR),
    ]

    serial_counts = count_encoded_exercises_figure_n_grams(
        samples,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=2,
        max_n=2,
        workers=1,
        batch_size=1,
    )
    parallel_counts = count_encoded_exercises_figure_n_grams(
        samples,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=2,
        max_n=2,
        workers=2,
        batch_size=1,
    )

    assert parallel_counts == serial_counts


def test_count_encoded_exercises_figure_ngrams_with_samples_keeps_sample_indices(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    first_tokens = _encoded_tokens(token_vocabulary, quarter_id=quarter_id)
    second_tokens = token_vocabulary.encode(
        [
            HandToken(hand=Hand.LEFT),
            _note(1, duration_id=quarter_id),
            _note(3, duration_id=quarter_id),
        ]
    )

    counted_figures = count_encoded_exercises_figure_ngrams_with_samples(
        [
            _sample(first_tokens, scale_type=ScaleType.MAJOR),
            _sample(second_tokens, scale_type=ScaleType.HARMONIC_MINOR),
        ],
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=2,
        max_n=2,
        workers=1,
        batch_size=1,
    )

    assert [sample_counts.sample_index for sample_counts in counted_figures.counts_by_sample] == [0, 1]
    assert [sample_counts.scale_type for sample_counts in counted_figures.counts_by_sample] == [
        ScaleType.MAJOR,
        ScaleType.HARMONIC_MINOR,
    ]
    assert sum(counted_figures.counts_by_sample[0].counts_by_hand[Hand.RIGHT][2].values()) == 1
    assert sum(counted_figures.counts_by_sample[1].counts_by_hand[Hand.LEFT][2].values()) == 1
    assert sum(counted_figures.counts_by_scale[ScaleType.MAJOR][Hand.RIGHT][2].values()) == 1
    assert sum(counted_figures.counts_by_scale[ScaleType.HARMONIC_MINOR][Hand.LEFT][2].values()) == 1


def _encoded_tokens(token_vocabulary: TokenVocabulary, *, quarter_id: int) -> list[int]:
    return token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, duration_id=quarter_id),
            _note(2, duration_id=quarter_id),
        ]
    )


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
    return NoteToken(degree=degree, accidental=0, octave_offset=0, duration_id=duration_id)
