from fractions import Fraction

import pytest

from musak_model.n_grams.figure.builder import (
    build_figure_ngram,
    build_figure_ngrams_from_run,
    build_figure_ngrams_from_runs,
)
from musak_model.n_grams.figure.parser import HandOnsetRun, PitchedOnset
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.tokens.schema import Hand, NoteToken, ScaleType, scale_size_for_type


def test_scale_size_comes_from_scale_type() -> None:
    assert scale_size_for_type(ScaleType.MAJOR) == 7
    assert scale_size_for_type(ScaleType.HARMONIC_MINOR) == 7
    assert scale_size_for_type(ScaleType.MELODIC_MINOR) == 7


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


def test_build_figure_ngrams_from_run_uses_inter_onset_duration_except_final_onset() -> None:
    run = HandOnsetRun(
        hand=Hand.RIGHT,
        onsets=(
            PitchedOnset(notes=(_note(1),), start=Fraction(0), duration=Fraction(1, 2)),
            PitchedOnset(notes=(_note(2),), start=Fraction(1, 8), duration=Fraction(1, 4)),
        ),
    )

    figures = build_figure_ngrams_from_run(run, n=2, scale_size=7)

    assert figures == (FigureNGram(onsets=((((0, 0),), Fraction(1, 1)), (((1, 0),), Fraction(2, 1)))),)


def test_build_figure_ngrams_from_run_slides_over_onset_run() -> None:
    run = HandOnsetRun(
        hand=Hand.RIGHT,
        onsets=(
            PitchedOnset(notes=(_note(1),), start=Fraction(0), duration=Fraction(1, 8)),
            PitchedOnset(notes=(_note(2),), start=Fraction(1, 8), duration=Fraction(1, 8)),
            PitchedOnset(notes=(_note(3),), start=Fraction(1, 4), duration=Fraction(1, 8)),
        ),
    )

    figures = build_figure_ngrams_from_run(run, n=2, scale_size=7)

    assert [figure.onsets for figure in figures] == [
        ((((0, 0),), Fraction(1)), (((1, 0),), Fraction(1))),
        ((((0, 0),), Fraction(1)), (((1, 0),), Fraction(1))),
    ]


def test_build_figure_ngrams_from_run_recomputes_duration_units_per_window() -> None:
    run = HandOnsetRun(
        hand=Hand.RIGHT,
        onsets=(
            PitchedOnset(notes=(_note(1),), start=Fraction(0), duration=Fraction(1, 2)),
            PitchedOnset(notes=(_note(2),), start=Fraction(1, 8), duration=Fraction(1, 2)),
            PitchedOnset(notes=(_note(3),), start=Fraction(1, 4), duration=Fraction(1, 8)),
        ),
    )

    two_grams = build_figure_ngrams_from_run(run, n=2, scale_size=7)
    three_grams = build_figure_ngrams_from_run(run, n=3, scale_size=7)

    assert [duration for _, duration in two_grams[0].onsets] == [Fraction(1), Fraction(4)]
    assert [duration for _, duration in three_grams[0].onsets] == [Fraction(1), Fraction(1), Fraction(1)]


def test_build_figure_ngrams_from_run_reanchors_later_windows() -> None:
    run = HandOnsetRun(
        hand=Hand.RIGHT,
        onsets=(
            PitchedOnset(notes=(_note(1),), start=Fraction(0), duration=Fraction(1, 8)),
            PitchedOnset(notes=(_note(3),), start=Fraction(1, 8), duration=Fraction(1, 8)),
            PitchedOnset(notes=(_note(4),), start=Fraction(1, 4), duration=Fraction(1, 8)),
        ),
    )

    figures = build_figure_ngrams_from_run(run, n=2, scale_size=7)

    assert figures[0].onsets[1][0] == ((2, 0),)
    assert figures[1].onsets[0][0] == ((0, 0),)
    assert figures[1].onsets[1][0] == ((1, 0),)


def test_build_figure_ngrams_from_runs_groups_by_figure_length() -> None:
    runs = [
        HandOnsetRun(
            hand=Hand.RIGHT,
            onsets=(
                PitchedOnset(notes=(_note(1),), start=Fraction(0), duration=Fraction(1, 8)),
                PitchedOnset(notes=(_note(2),), start=Fraction(1, 8), duration=Fraction(1, 8)),
                PitchedOnset(notes=(_note(3),), start=Fraction(1, 4), duration=Fraction(1, 8)),
            ),
        )
    ]

    figures_by_length = build_figure_ngrams_from_runs(runs, min_n=2, max_n=3, scale_size=7)

    assert len(figures_by_length[2]) == 2
    assert len(figures_by_length[3]) == 1


def test_build_figure_ngrams_from_run_rejects_non_positive_figure_length() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        build_figure_ngrams_from_run(
            HandOnsetRun(hand=Hand.RIGHT, onsets=()),
            n=0,
            scale_size=7,
        )


def _note(degree: int) -> NoteToken:
    return NoteToken(degree=degree, accidental=0, octave_offset=0, duration_id=0)
