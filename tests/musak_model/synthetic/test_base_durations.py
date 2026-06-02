from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import fmean

import polars as pl
import pytest
from numpy.random import default_rng

from musak_model.n_grams.figure.schema import FigureNGram
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
    choose_base_duration,
    fitting_base_durations,
    load_base_duration_distribution,
    weighted_base_duration_choice,
)
from musak_model.tokens.duration import DurationVocabulary
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


_CANDIDATES = ((Fraction(1, 16), 1), (Fraction(1, 8), 3), (Fraction(1, 4), 1))


def _figure(*steps: int) -> FigureNGram:
    return FigureNGram(onsets=tuple((((step, 0),), Fraction(1)) for step in steps))


def test_fitting_excludes_bases_that_overrun_the_remaining_space(duration_vocabulary: DurationVocabulary) -> None:
    fitting = fitting_base_durations(
        _figure(0, 1), _CANDIDATES, remaining=Fraction(1, 32), duration_vocabulary=duration_vocabulary
    )

    assert fitting == []


def test_choose_base_duration_centers_on_the_weighted_median(duration_vocabulary: DurationVocabulary) -> None:
    base = choose_base_duration(
        _figure(0, 1), _CANDIDATES, density_offset=0.0, remaining=Fraction(1), duration_vocabulary=duration_vocabulary
    )

    assert base == Fraction(1, 8)


def test_density_offset_lengthens_or_shortens_the_base(duration_vocabulary: DurationVocabulary) -> None:
    longer = choose_base_duration(
        _figure(0, 1), _CANDIDATES, density_offset=1.0, remaining=Fraction(1), duration_vocabulary=duration_vocabulary
    )
    shorter = choose_base_duration(
        _figure(0, 1), _CANDIDATES, density_offset=-1.0, remaining=Fraction(1), duration_vocabulary=duration_vocabulary
    )

    assert longer == Fraction(1, 4)
    assert shorter == Fraction(1, 16)


def test_choose_base_duration_returns_none_when_nothing_fits(duration_vocabulary: DurationVocabulary) -> None:
    base = choose_base_duration(
        _figure(0, 1),
        _CANDIDATES,
        density_offset=0.0,
        remaining=Fraction(1, 32),
        duration_vocabulary=duration_vocabulary,
    )

    assert base is None
