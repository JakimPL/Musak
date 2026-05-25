from fractions import Fraction

import pytest
from pydantic import ValidationError

from musak_model.analysis.n_grams.figure.builder import (
    build_figure_ngram,
    build_figure_ngrams_from_run,
    build_figure_ngrams_from_runs,
    note_diatonic_position,
    scale_size_for_type,
)
from musak_model.analysis.n_grams.figure.parser import HandOnsetRun, PitchedOnset
from musak_model.analysis.n_grams.figure.schema import FigureNGram
from musak_model.tokens.schema import Hand, NoteToken, ScaleType


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


def test_figure_ngram_string_represents_relative_degrees_and_normalized_durations() -> None:
    figure = FigureNGram(
        onsets=(
            (((0, 0), (2, -1), (4, 1)), Fraction(1)),
            (((-1, 0),), Fraction(3, 2)),
        )
    )

    assert str(figure) == "[0 +2b +4#](1) -1(3/2)"
    assert repr(figure) == "FigureNGram('[0 +2b +4#](1) -1(3/2)')"


def test_figure_ngram_properties_classify_texture_and_accidentals() -> None:
    monophonic = FigureNGram(onsets=((((0, 0),), Fraction(1)), (((2, 0),), Fraction(1))))
    chords_only = FigureNGram(onsets=((((0, 0), (2, 0)), Fraction(1)), (((1, 0), (3, 0)), Fraction(1))))
    mixed = FigureNGram(onsets=((((0, 0), (2, 0)), Fraction(1)), (((1, 1),), Fraction(1))))

    assert monophonic.monophonic
    assert not monophonic.chords_only
    assert monophonic.in_scale

    assert not chords_only.monophonic
    assert chords_only.chords_only
    assert chords_only.in_scale

    assert not mixed.monophonic
    assert not mixed.chords_only
    assert not mixed.in_scale


def test_build_figure_ngrams_from_run_uses_inter_onset_duration_except_final_onset() -> None:
    run = HandOnsetRun(
        hand=Hand.RIGHT,
        onsets=(
            PitchedOnset(
                notes=(_note(1),),
                start=Fraction(0),
                duration=Fraction(1, 2),
            ),
            PitchedOnset(
                notes=(_note(2),),
                start=Fraction(1, 8),
                duration=Fraction(1, 4),
            ),
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


def test_build_figure_ngrams_from_runs_groups_by_n() -> None:
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

    figures_by_n = build_figure_ngrams_from_runs(runs, min_n=2, max_n=3, scale_size=7)

    assert len(figures_by_n[2]) == 2
    assert len(figures_by_n[3]) == 1


def test_build_figure_ngrams_from_run_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        build_figure_ngrams_from_run(
            HandOnsetRun(hand=Hand.RIGHT, onsets=()),
            n=0,
            scale_size=7,
        )


def _note(degree: int) -> NoteToken:
    return NoteToken(
        degree=degree,
        accidental=0,
        octave_offset=0,
        duration_id=0,
    )
