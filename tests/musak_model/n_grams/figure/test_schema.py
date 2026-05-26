from fractions import Fraction

import pytest
from pydantic import ValidationError

from musak_model.n_grams.figure.schema import FigureNGram


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
