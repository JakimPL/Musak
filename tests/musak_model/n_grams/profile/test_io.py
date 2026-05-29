from collections import Counter
from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.builder import build_figure_profile, build_figure_sample_counts
from musak_model.n_grams.profile.io import (
    FIGURE_COUNT_SCHEMA,
    figure_counts_frame,
    read_figure_counts,
    read_figure_counts_for_groups,
    read_figure_profile,
    read_figure_sample_counts_jsonl,
    write_figure_counts,
    write_figure_profile,
    write_figure_sample_counts_jsonl,
)
from musak_model.n_grams.profile.schema import FigureProfileMetadata
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.tables import write_table


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


def test_figure_counts_round_trips(tmp_path: Path) -> None:
    figure = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    counts = {
        ScaleType.MAJOR: {
            Hand.RIGHT: {
                1: Counter({figure: 2}),
            }
        }
    }
    path = tmp_path / "counts.parquet"

    write_figure_counts(counts, path)

    assert read_figure_counts(path) == counts


def test_filtered_figure_counts_skips_unrelated_rows_before_parsing_json(tmp_path: Path) -> None:
    figure = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    path = tmp_path / "counts.parquet"
    write_table(
        pl.DataFrame(
            [
                {"scale_type": "major", "hand": "right", "n": 2, "count": 3, "figure": figure.model_dump_json()},
                {
                    "scale_type": "harmonic_minor",
                    "hand": "right",
                    "n": 2,
                    "count": 1,
                    "figure": "not valid figure json",
                },
                {"scale_type": "major", "hand": "left", "n": 3, "count": 1, "figure": "not valid figure json"},
            ],
            schema=FIGURE_COUNT_SCHEMA,
            orient="row",
        ),
        path,
    )

    assert read_figure_counts_for_groups(
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


def test_figure_counts_frame_serializes_counts_in_stable_order() -> None:
    figure = FigureNGram(onsets=((((0, 0),), Fraction(1)),))

    frame = figure_counts_frame({ScaleType.MAJOR: {Hand.RIGHT: {1: Counter({figure: 2})}}})

    assert frame.to_dicts() == [
        {"scale_type": "major", "hand": "right", "n": 1, "count": 2, "figure": '{"onsets":[[[[0,0]],"1"]]}'}
    ]
    assert frame.columns == list(FIGURE_COUNT_SCHEMA)


def test_figure_counts_frame_limits_each_group() -> None:
    first = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    second = FigureNGram(onsets=((((1, 0),), Fraction(1)),))

    frame = figure_counts_frame(
        {ScaleType.MAJOR: {Hand.RIGHT: {1: Counter({first: 2, second: 1})}}},
        limit_per_group=1,
    )

    assert frame.height == 1
    assert frame["count"].to_list() == [2]


def test_figure_counts_frame_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit_per_group must be positive"):
        figure_counts_frame({}, limit_per_group=0)
