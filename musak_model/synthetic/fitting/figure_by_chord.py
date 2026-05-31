from __future__ import annotations

from collections import defaultdict
from math import log
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.artifacts import FIGURE_ALL_DIR_NAME, FIGURE_DIR_NAME
from musak_model.n_grams.profile.chord.schema import FigureByChordCounts, chord_from_key
from musak_model.synthetic.substitution.chord_figure import FigureByChordKey, FigureByChordModel, FigureByChordTable
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.tables import read_table, write_table

FITTED_FIGURE_BY_CHORD_NAME: Final[str] = "figure_by_chord.parquet"

_SCALE_TYPE_COLUMN: Final[str] = "scale_type"
_HAND_COLUMN: Final[str] = "hand"
_N_COLUMN: Final[str] = "n"
_CHORD_COLUMN: Final[str] = "chord"
_FIGURE_COLUMN: Final[str] = "figure"
_LOG_PROBABILITY_COLUMN: Final[str] = "log_probability"

FITTED_FIGURE_BY_CHORD_SCHEMA: Final[dict[str, pl.DataType]] = {
    _SCALE_TYPE_COLUMN: pl.String(),
    _HAND_COLUMN: pl.String(),
    _N_COLUMN: pl.Int64(),
    _CHORD_COLUMN: pl.String(),
    _FIGURE_COLUMN: pl.String(),
    _LOG_PROBABILITY_COLUMN: pl.Float64(),
}


class FittedFigureByChordRow(NamedTuple):
    scale_type: str
    hand: str
    figure_length: int
    chord: str
    figure: str
    log_probability: float


def fit_figure_by_chord_rows(counts: FigureByChordCounts, *, limit: int) -> list[FittedFigureByChordRow]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    figures_by_group: dict[tuple[str, str, int, str], list[tuple[str, int]]] = defaultdict(list)
    totals: dict[tuple[str, str, int, str], int] = defaultdict(int)
    for key, count in counts.items():
        group = (key.scale_type, key.hand, key.figure_length, key.chord)
        figures_by_group[group].append((key.figure, count))
        totals[group] += count

    rows: list[FittedFigureByChordRow] = []
    for group, figure_counts in figures_by_group.items():
        total = totals[group]
        scale_type, hand, figure_length, chord = group
        for figure, count in sorted(figure_counts, key=lambda item: (-item[1], item[0]))[:limit]:
            rows.append(FittedFigureByChordRow(scale_type, hand, figure_length, chord, figure, log(count / total)))

    return rows


def write_figure_by_chord_table(rows: list[FittedFigureByChordRow], path: Path) -> None:
    records = [
        {
            _SCALE_TYPE_COLUMN: row.scale_type,
            _HAND_COLUMN: row.hand,
            _N_COLUMN: row.figure_length,
            _CHORD_COLUMN: row.chord,
            _FIGURE_COLUMN: row.figure,
            _LOG_PROBABILITY_COLUMN: row.log_probability,
        }
        for row in rows
    ]
    write_table(pl.DataFrame(records, schema=FITTED_FIGURE_BY_CHORD_SCHEMA, orient="row"), path)


def load_figure_by_chord_model(figure_directory: Path) -> FigureByChordModel:
    path = resolve_fitted_figure_by_chord_path(figure_directory)
    if path is None:
        return FigureByChordModel()

    grouped: dict[FigureByChordKey, dict[FigureNGram, float]] = defaultdict(dict)
    for row in read_table(path).iter_rows(named=True):
        key = (
            ScaleType(row[_SCALE_TYPE_COLUMN]),
            Hand(row[_HAND_COLUMN]),
            int(row[_N_COLUMN]),
            chord_from_key(row[_CHORD_COLUMN]),
        )
        grouped[key][FigureNGram.model_validate_json(row[_FIGURE_COLUMN])] = float(row[_LOG_PROBABILITY_COLUMN])

    tables = {
        key: FigureByChordTable(log_probabilities=log_probabilities, floor=min(log_probabilities.values()))
        for key, log_probabilities in grouped.items()
    }
    return FigureByChordModel(tables=tables)


def resolve_fitted_figure_by_chord_path(path: Path) -> Path | None:
    if path.is_file():
        return path

    candidates = (
        path / FITTED_FIGURE_BY_CHORD_NAME,
        path / FIGURE_ALL_DIR_NAME / FITTED_FIGURE_BY_CHORD_NAME,
        path / FIGURE_DIR_NAME / FIGURE_ALL_DIR_NAME / FITTED_FIGURE_BY_CHORD_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None
