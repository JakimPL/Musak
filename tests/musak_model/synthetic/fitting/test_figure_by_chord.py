from collections import Counter
from fractions import Fraction
from math import log
from pathlib import Path

import pytest

from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.chord.schema import FigureByChordCountKey, FigureByChordCounts, chord_to_key
from musak_model.synthetic.fitting.figure_by_chord import (
    FITTED_FIGURE_BY_CHORD_NAME,
    fit_figure_by_chord_rows,
    load_figure_by_chord_model,
    write_figure_by_chord_table,
)
from musak_model.tokens.schema import Hand, ScaleType

_TONIC = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)


def _figure(positions: list[int]) -> FigureNGram:
    return FigureNGram(onsets=tuple((((position, 0),), Fraction(1)) for position in positions))


def _counts(figure_counts: list[tuple[FigureNGram, int]]) -> FigureByChordCounts:
    return Counter(
        {
            FigureByChordCountKey("major", "right", 2, chord_to_key(_TONIC), figure.model_dump_json()): count
            for figure, count in figure_counts
        }
    )


def test_fit_rows_prune_to_top_k_and_normalize_over_full_total() -> None:
    common = _figure([0, 2])
    medium = _figure([0, 1])
    rare = _figure([0, -2])
    rows = fit_figure_by_chord_rows(_counts([(common, 6), (medium, 3), (rare, 1)]), limit=2)

    by_figure = {row.figure: row.log_probability for row in rows}

    assert len(rows) == 2  # the rare figure is dropped by the top-2 cap
    assert by_figure[common.model_dump_json()] == pytest.approx(log(0.6))  # normalized over the full total (10)
    assert by_figure[medium.model_dump_json()] == pytest.approx(log(0.3))


def test_write_and_load_round_trips_to_model_with_precomputed_floor(tmp_path: Path) -> None:
    common = _figure([0, 2])
    medium = _figure([0, 1])
    rows = fit_figure_by_chord_rows(_counts([(common, 6), (medium, 3), (_figure([0, -2]), 1)]), limit=2)
    write_figure_by_chord_table(rows, tmp_path / FITTED_FIGURE_BY_CHORD_NAME)

    model = load_figure_by_chord_model(tmp_path)
    table = model.table(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2, chord=_TONIC)

    assert table is not None
    assert table.log_probabilities[common] == pytest.approx(log(0.6))
    assert table.floor == pytest.approx(log(0.3))  # the least-likely kept entry


def test_load_returns_empty_model_when_artifact_absent(tmp_path: Path) -> None:
    assert load_figure_by_chord_model(tmp_path / "missing").tables == {}
