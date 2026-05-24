from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.analysis.n_grams import (
    COUNT_CSV_COLUMNS,
    FigureNGram,
    figure_count_records,
    write_figure_count_csv,
)
from musak_model.tokens.schema import Hand, ScaleType


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
