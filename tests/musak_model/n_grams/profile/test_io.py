from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.builder import build_figure_profile, build_figure_sample_counts
from musak_model.n_grams.profile.io import (
    COUNT_CSV_COLUMNS,
    figure_count_records,
    read_figure_counts_csv,
    read_figure_counts_csv_for_groups,
    read_figure_profile,
    read_figure_sample_counts_jsonl,
    write_figure_count_csv,
    write_figure_counts_csv,
    write_figure_profile,
    write_figure_sample_counts_jsonl,
)
from musak_model.n_grams.profile.schema import FigureProfileMetadata
from musak_model.tokens.schema import Hand, ScaleType


def test_figure_profile_json_round_trips(tmp_path: Path) -> None:
    profile = build_figure_profile(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)),)): 2}),
                }
            }
        },
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=3),
    )
    path = tmp_path / "profile.json"

    write_figure_profile(profile, path)

    assert read_figure_profile(path) == profile


def test_figure_counts_csv_round_trips(tmp_path: Path) -> None:
    figure = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    counts = {
        ScaleType.MAJOR: {
            Hand.RIGHT: {
                1: Counter({figure: 2}),
            }
        }
    }
    path = tmp_path / "counts.csv"

    write_figure_counts_csv(counts, path)

    assert read_figure_counts_csv(path) == counts


def test_filtered_figure_counts_csv_skips_unrelated_rows_before_parsing_json(tmp_path: Path) -> None:
    figure = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    path = tmp_path / "counts.csv"
    write_figure_count_csv(
        [
            {
                "scale_type": "major",
                "hand": "right",
                "n": 2,
                "count": 3,
                "figure": figure.model_dump_json(),
            },
            {
                "scale_type": "harmonic_minor",
                "hand": "right",
                "n": 2,
                "count": 1,
                "figure": "not valid figure json",
            },
            {
                "scale_type": "major",
                "hand": "left",
                "n": 3,
                "count": 1,
                "figure": "not valid figure json",
            },
        ],
        path,
    )

    assert read_figure_counts_csv_for_groups(
        path,
        scale_type=ScaleType.MAJOR,
        groups=frozenset({(Hand.RIGHT, 2)}),
    ) == {ScaleType.MAJOR: {Hand.RIGHT: {2: Counter({figure: 3})}}}


def test_figure_sample_counts_jsonl_round_trips(tmp_path: Path) -> None:
    sample_counts = build_figure_sample_counts(
        sample_index=3,
        scale_type=ScaleType.MAJOR,
        counts_by_hand={
            Hand.RIGHT: {
                1: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)),)): 2}),
            }
        },
    )
    path = tmp_path / "by_sample.jsonl"

    write_figure_sample_counts_jsonl([sample_counts], path)

    assert read_figure_sample_counts_jsonl(path) == [sample_counts]


def test_figure_count_records_serializes_counts_in_stable_order() -> None:
    figure = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    records = figure_count_records(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter({figure: 2}),
                }
            }
        }
    )

    assert records == [
        {
            "scale_type": "major",
            "hand": "right",
            "n": 1,
            "count": 2,
            "figure": '{"onsets":[[[[0,0]],"1"]]}',
        }
    ]


def test_figure_count_records_limits_each_group() -> None:
    first = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    second = FigureNGram(onsets=((((1, 0),), Fraction(1)),))
    records = figure_count_records(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter({first: 2, second: 1}),
                }
            }
        },
        limit_per_group=1,
    )

    assert len(records) == 1
    assert records[0]["count"] == 2


def test_figure_count_records_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit_per_group must be positive"):
        figure_count_records({}, limit_per_group=0)


def test_write_figure_count_csv(tmp_path: Path) -> None:
    path = tmp_path / "figures.csv"
    write_figure_count_csv(
        [
            {
                "scale_type": "major",
                "hand": "right",
                "n": 1,
                "count": 2,
                "figure": '{"onsets":[[[[0,0]],"1"]]}',
            }
        ],
        path,
    )

    assert path.read_text(encoding="utf-8").splitlines() == [
        ",".join(COUNT_CSV_COLUMNS),
        'major,right,1,2,"{""onsets"":[[[[0,0]],""1""]]}"',
    ]
