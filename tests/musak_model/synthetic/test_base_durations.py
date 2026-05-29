from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import fmean

import polars as pl
import pytest
from numpy.random import default_rng

from musak_model.n_grams.profile.io import (
    BASE_DURATION_COLUMN,
    BASE_DURATION_SCHEMA,
    COUNT_COLUMN,
    HAND_COLUMN,
    N_COLUMN,
    SCALE_TYPE_COLUMN,
)
from musak_model.synthetic.base_durations import (
    BaseDurationDistribution,
    load_base_duration_distribution,
    weighted_base_duration_choice,
)
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.tables import write_table


def _write_base_durations(path: Path, rows: list[tuple[ScaleType, Hand, int, Fraction, int]]) -> None:
    records = [
        {
            SCALE_TYPE_COLUMN: scale_type.value,
            HAND_COLUMN: hand.value,
            N_COLUMN: figure_length,
            BASE_DURATION_COLUMN: f"{base_duration.numerator}/{base_duration.denominator}",
            COUNT_COLUMN: count,
        }
        for scale_type, hand, figure_length, base_duration, count in rows
    ]
    write_table(pl.DataFrame(records, schema=BASE_DURATION_SCHEMA, orient="row"), path)


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


def test_load_base_duration_distribution_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "base_durations.parquet"
    _write_base_durations(
        path,
        [
            (ScaleType.MAJOR, Hand.RIGHT, 2, Fraction(1, 8), 3),
            (ScaleType.MAJOR, Hand.RIGHT, 2, Fraction(1, 4), 1),
            (ScaleType.MAJOR, Hand.LEFT, 3, Fraction(1, 16), 5),
        ],
    )

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
    _write_base_durations(
        all_directory / "base_durations.parquet", [(ScaleType.MAJOR, Hand.RIGHT, 2, Fraction(1, 4), 1)]
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
