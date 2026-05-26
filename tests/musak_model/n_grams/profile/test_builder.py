from collections import Counter
from fractions import Fraction

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.builder import build_figure_profile
from musak_model.n_grams.profile.schema import FigureProfileMetadata
from musak_model.tokens.schema import Hand, ScaleType


def test_build_figure_profile_totals_are_deterministic() -> None:
    first = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    second = FigureNGram(onsets=((((0, 0), (2, 0)), Fraction(1)),))

    profile = build_figure_profile(
        {
            ScaleType.HARMONIC_MINOR: {Hand.RIGHT: {1: Counter({first: 2})}},
            ScaleType.MAJOR: {Hand.LEFT: {1: Counter({second: 3})}},
        },
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=5),
    )

    assert [(group.scale_type, group.hand, group.n, group.total) for group in profile.groups] == [
        (ScaleType.HARMONIC_MINOR, Hand.RIGHT, 1, 2),
        (ScaleType.MAJOR, Hand.LEFT, 1, 3),
    ]


def test_build_figure_profile_property_totals_are_count_weighted() -> None:
    monophonic = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    chord = FigureNGram(onsets=((((0, 0), (2, 0)), Fraction(1)),))
    chromatic = FigureNGram(onsets=((((0, 1),), Fraction(1)),))

    profile = build_figure_profile(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter(
                        {
                            monophonic: 2,
                            chord: 3,
                            chromatic: 5,
                        }
                    )
                }
            }
        },
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=1),
    )

    assert len(profile.groups) == 1
    group = profile.groups[0]
    assert group.total == 10
    assert group.monophonic == 7
    assert group.chords_only == 3
    assert group.in_scale == 5
