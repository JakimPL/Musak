from fractions import Fraction

import pytest
from pydantic import ValidationError

from musak_model.analysis.n_grams import (
    FigureNGram,
    build_figure_ngram,
    note_diatonic_position,
    scale_size_for_type,
)
from musak_model.tokens.schema import NoteToken, ScaleType


def test_scale_size_comes_from_scale_type() -> None:
    assert scale_size_for_type(ScaleType.MAJOR) == 7
    assert scale_size_for_type(ScaleType.HARMONIC_MINOR) == 7
    assert scale_size_for_type(ScaleType.MELODIC_MINOR) == 7


def test_note_diatonic_position_uses_configured_scale_size() -> None:
    token = NoteToken(degree=3, accidental=0, octave_offset=2, duration_id=0)

    assert note_diatonic_position(token, scale_size=7) == 16
    assert note_diatonic_position(token, scale_size=5) == 12


def test_build_figure_ngram_is_transposition_invariant_in_scale_degrees() -> None:
    first = build_figure_ngram(
        [
            [NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0)],
            [NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=0)],
        ],
        [Fraction(1, 8), Fraction(1, 8)],
        scale_size=7,
    )
    transposed = build_figure_ngram(
        [
            [NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=0)],
            [NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=0)],
        ],
        [Fraction(1, 4), Fraction(1, 4)],
        scale_size=7,
    )

    assert transposed == first
    assert first.onsets == ((((0, 0),), Fraction(1)), (((2, 0),), Fraction(1)))


def test_build_figure_ngram_preserves_accidentals_on_relative_degrees() -> None:
    figure = build_figure_ngram(
        [
            [NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0)],
            [NoteToken(degree=4, accidental=1, octave_offset=0, duration_id=0)],
        ],
        [Fraction(1, 8), Fraction(1, 8)],
        scale_size=7,
    )

    assert figure.onsets[1][0] == ((3, 1),)


def test_build_figure_ngram_normalizes_durations_by_shortest_duration() -> None:
    figure = build_figure_ngram(
        [
            [NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0)],
            [NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=0)],
            [NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=0)],
        ],
        [Fraction(1, 4), Fraction(1, 8), Fraction(3, 8)],
        scale_size=7,
    )

    assert [onset[1] for onset in figure.onsets] == [Fraction(2), Fraction(1), Fraction(3)]


def test_build_figure_ngram_sorts_chord_pitches() -> None:
    figure = build_figure_ngram(
        [
            [
                NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=0),
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0),
                NoteToken(degree=3, accidental=-1, octave_offset=0, duration_id=0),
            ],
        ],
        [Fraction(1, 4)],
        scale_size=7,
    )

    assert figure.onsets == ((((0, 0), (2, -1), (4, 0)), Fraction(1)),)


def test_build_figure_ngram_rejects_empty_onsets() -> None:
    with pytest.raises(ValueError, match="every onset"):
        build_figure_ngram([[]], [Fraction(1, 4)], scale_size=7)


def test_build_figure_ngram_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="durations must be positive"):
        build_figure_ngram(
            [[NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0)]],
            [Fraction(0)],
            scale_size=7,
        )


def test_figure_ngram_requires_at_least_one_onset() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        FigureNGram(onsets=())
