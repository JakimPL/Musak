from fractions import Fraction

import pytest

from musak_model.analysis.n_grams.figure.counter import count_figure_ngrams, count_hand_figure_ngrams
from musak_model.analysis.n_grams.figure.parser import HandOnsetRun, PitchedOnset
from musak_model.tokens.schema import Hand, NoteToken


def test_count_figure_ngrams_groups_counts_by_n() -> None:
    counts = count_figure_ngrams(
        [
            HandOnsetRun(
                hand=Hand.RIGHT,
                onsets=(
                    PitchedOnset(notes=(_note(1),), start=Fraction(0), duration=Fraction(1, 8)),
                    PitchedOnset(notes=(_note(2),), start=Fraction(1, 8), duration=Fraction(1, 8)),
                    PitchedOnset(notes=(_note(3),), start=Fraction(1, 4), duration=Fraction(1, 8)),
                ),
            )
        ],
        min_n=2,
        max_n=3,
        scale_size=7,
    )

    assert sum(counts[2].values()) == 2
    assert sum(counts[3].values()) == 1
    assert counts[2].most_common(1)[0][1] == 2


def test_count_figure_ngrams_counts_across_runs_without_crossing_boundaries() -> None:
    counts = count_figure_ngrams(
        [
            HandOnsetRun(
                hand=Hand.RIGHT,
                onsets=(
                    PitchedOnset(notes=(_note(1),), start=Fraction(0), duration=Fraction(1, 8)),
                    PitchedOnset(notes=(_note(2),), start=Fraction(1, 8), duration=Fraction(1, 8)),
                ),
            ),
            HandOnsetRun(
                hand=Hand.RIGHT,
                onsets=(PitchedOnset(notes=(_note(3),), start=Fraction(1, 2), duration=Fraction(1, 8)),),
            ),
        ],
        min_n=2,
        max_n=2,
        scale_size=7,
    )

    assert sum(counts[2].values()) == 1


def test_count_hand_figure_ngrams_keeps_hands_separate() -> None:
    counts = count_hand_figure_ngrams(
        {
            Hand.RIGHT: (
                HandOnsetRun(
                    hand=Hand.RIGHT,
                    onsets=(
                        PitchedOnset(notes=(_note(1),), start=Fraction(0), duration=Fraction(1, 8)),
                        PitchedOnset(notes=(_note(2),), start=Fraction(1, 8), duration=Fraction(1, 8)),
                    ),
                ),
            ),
            Hand.LEFT: (),
        },
        min_n=2,
        max_n=2,
        scale_size=7,
    )

    assert sum(counts[Hand.RIGHT][2].values()) == 1
    assert sum(counts[Hand.LEFT][2].values()) == 0


def test_count_figure_ngrams_rejects_invalid_n_range() -> None:
    with pytest.raises(ValueError, match="max_n must be greater than or equal to min_n"):
        count_figure_ngrams((), min_n=3, max_n=2, scale_size=7)


def _note(degree: int) -> NoteToken:
    return NoteToken(
        degree=degree,
        accidental=0,
        octave_offset=0,
        duration_id=0,
    )
