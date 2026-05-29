from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import fmean

import pytest
from numpy.random import default_rng

from musak_model.n_grams.profile.io import BASE_DURATION_CSV_COLUMNS
from musak_model.synthetic.base_durations import (
    BaseDurationDistribution,
    load_base_duration_distribution,
    weighted_base_duration_choice,
)
from musak_model.tokens.schema import Hand, ScaleType


def _distribution() -> BaseDurationDistribution:
    return BaseDurationDistribution(
        weights_by_group={
            (ScaleType.MAJOR, Hand.RIGHT, 2): ((Fraction(1, 8), 3), (Fraction(1, 4), 1)),
        }
    )


def test_candidates_returns_group_weights() -> None:
    distribution = _distribution()

    assert distribution.candidates(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2) == (
        (Fraction(1, 8), 3),
        (Fraction(1, 4), 1),
    )
    assert distribution.candidates(scale_type=ScaleType.MAJOR, hand=Hand.LEFT, figure_length=2) == ()


def test_sample_rejects_empty_group() -> None:
    distribution = _distribution()

    with pytest.raises(ValueError, match="no base durations"):
        distribution.sample(scale_type=ScaleType.MAJOR, hand=Hand.LEFT, figure_length=2, rng=default_rng(0))


def test_sample_follows_weights() -> None:
    distribution = _distribution()
    rng = default_rng(0)

    draws = [
        distribution.sample(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2, rng=rng) for _ in range(4000)
    ]

    eighth_rate = fmean(1.0 if draw == Fraction(1, 8) else 0.0 for draw in draws)
    assert abs(eighth_rate - 0.75) < 0.03


def test_weighted_choice_rejects_empty() -> None:
    with pytest.raises(ValueError, match="candidates must be non-empty"):
        weighted_base_duration_choice((), rng=default_rng(0))


def test_load_base_duration_distribution_round_trips_csv(tmp_path: Path) -> None:
    path = tmp_path / "base_durations.csv"
    rows = [
        (ScaleType.MAJOR, Hand.RIGHT, 2, Fraction(1, 8), 3),
        (ScaleType.MAJOR, Hand.RIGHT, 2, Fraction(1, 4), 1),
        (ScaleType.MAJOR, Hand.LEFT, 3, Fraction(1, 16), 5),
    ]
    lines = [",".join(BASE_DURATION_CSV_COLUMNS)]
    for scale_type, hand, figure_length, base_duration, count in rows:
        lines.append(
            f"{scale_type.value},{hand.value},{figure_length},{base_duration.numerator}/{base_duration.denominator},{count}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    distribution = load_base_duration_distribution(path)

    assert distribution.candidates(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2) == (
        (Fraction(1, 8), 3),
        (Fraction(1, 4), 1),
    )
    assert distribution.candidates(scale_type=ScaleType.MAJOR, hand=Hand.LEFT, figure_length=3) == (
        (Fraction(1, 16), 5),
    )


def test_load_base_duration_distribution_resolves_artifact_directory(tmp_path: Path) -> None:
    all_directory = tmp_path / "figure" / "all"
    all_directory.mkdir(parents=True)
    path = all_directory / "base_durations.csv"
    path.write_text(
        "\n".join([",".join(BASE_DURATION_CSV_COLUMNS), "major,right,2,1/4,1"]) + "\n",
        encoding="utf-8",
    )

    distribution = load_base_duration_distribution(tmp_path)

    assert distribution.candidates(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2) == (
        (Fraction(1, 4), 1),
    )


def test_aggregated_counter_round_trip_matches_expected() -> None:
    distribution = BaseDurationDistribution(
        weights_by_group={(ScaleType.MAJOR, Hand.RIGHT, 2): tuple(sorted(Counter({Fraction(1, 4): 2}).items()))}
    )

    assert distribution.candidates(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2) == (
        (Fraction(1, 4), 2),
    )
