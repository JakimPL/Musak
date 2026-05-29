from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Final

import polars as pl

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.io import (
    COUNT_COLUMN,
    FIGURE_COLUMN,
    HAND_COLUMN,
    N_COLUMN,
    SCALE_TYPE_COLUMN,
)
from musak_model.paths import DEFAULT_ANALYSIS_DIR
from musak_shared.notation.schema import (
    EIGHTH,
    HALF,
    QUARTER,
    SIXTEENTH,
    THIRTY_SECOND,
    WHOLE,
    NoteData,
    ScoreData,
    StaveData,
    VexflowAccidental,
    VexflowDuration,
    VoiceData,
)
from musak_shared.tables import read_table

FIGURE_PERCENT_COLUMN: Final[str] = "percent"
FIGURE_TOTAL_COLUMN: Final[str] = "total_count"
FIGURE_UNIQUE_COLUMN: Final[str] = "unique_figures"
FIGURE_DURATION_UNIT_COLUMN: Final[str] = "duration_unit"
FIGURE_LABEL_COLUMN: Final[str] = "label"
FIGURE_TEXT_COLUMN: Final[str] = "figure_text"
FIGURE_MONOPHONIC_COLUMN: Final[str] = "monophonic"
FIGURE_CHORDS_ONLY_COLUMN: Final[str] = "chords_only"
FIGURE_IN_SCALE_COLUMN: Final[str] = "in_scale"
FIGURE_PROPERTY_COLUMN: Final[str] = "property"
FIGURE_PROPERTY_VALUE_COLUMN: Final[str] = "value"

_FIGURE_COUNT_COLUMNS: Final[frozenset[str]] = frozenset(
    {SCALE_TYPE_COLUMN, HAND_COLUMN, N_COLUMN, COUNT_COLUMN, FIGURE_COLUMN}
)
_LETTER_NAMES: Final[tuple[str, ...]] = ("c", "d", "e", "f", "g", "a", "b")
_ACCIDENTAL_BY_OFFSET: Final[dict[int, VexflowAccidental]] = {-1: "b", 0: None, 1: "#"}
_POWER_DURATION_BY_FRACTION: Final[dict[Fraction, VexflowDuration]] = {
    Fraction(1, 1): WHOLE,
    Fraction(1, 2): HALF,
    Fraction(1, 4): QUARTER,
    Fraction(1, 8): EIGHTH,
    Fraction(1, 16): SIXTEENTH,
    Fraction(1, 32): THIRTY_SECOND,
}
_MAX_DOTS: Final[int] = 2
_DEFAULT_PREFERRED_UNIT: Final[Fraction] = Fraction(1, 8)
_DISPLAY_UNIT_CANDIDATES: Final[tuple[Fraction, ...]] = (
    Fraction(1, 8),
    Fraction(1, 16),
    Fraction(1, 32),
    Fraction(1, 4),
    Fraction(1, 2),
    Fraction(1, 1),
)
_REPRESENTABLE_DURATIONS: Final[dict[Fraction, tuple[VexflowDuration, int]]] = {
    duration * Fraction(2 ** (dots + 1) - 1, 2**dots): (vexflow_duration, dots)
    for duration, vexflow_duration in _POWER_DURATION_BY_FRACTION.items()
    for dots in range(_MAX_DOTS + 1)
    if duration * Fraction(2 ** (dots + 1) - 1, 2**dots) <= 1
}
_FIGURE_PROPERTY_COLUMNS: Final[tuple[str, ...]] = (
    FIGURE_MONOPHONIC_COLUMN,
    FIGURE_CHORDS_ONLY_COLUMN,
    FIGURE_IN_SCALE_COLUMN,
)


def analysis_result_files(analysis_dir: Path = DEFAULT_ANALYSIS_DIR) -> list[Path]:
    if not analysis_dir.exists():
        return []

    return sorted(path for path in analysis_dir.glob("*.parquet") if path.is_file())


def read_figure_count_frame(path: Path) -> pl.DataFrame:
    frame = read_table(path)
    _require_figure_count_columns(frame)
    return frame.with_columns(
        pl.col(N_COLUMN).cast(pl.Int64),
        pl.col(COUNT_COLUMN).cast(pl.Int64),
    )


def figure_group_summary(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                SCALE_TYPE_COLUMN: pl.String(),
                HAND_COLUMN: pl.String(),
                N_COLUMN: pl.Int64(),
                FIGURE_TOTAL_COLUMN: pl.Int64(),
                FIGURE_UNIQUE_COLUMN: pl.UInt32(),
            }
        )

    return (
        frame.group_by([SCALE_TYPE_COLUMN, HAND_COLUMN, N_COLUMN])
        .agg(
            pl.col(COUNT_COLUMN).sum().alias(FIGURE_TOTAL_COLUMN),
            pl.col(FIGURE_COLUMN).len().alias(FIGURE_UNIQUE_COLUMN),
        )
        .sort([SCALE_TYPE_COLUMN, HAND_COLUMN, N_COLUMN])
    )


def top_figure_frame(
    frame: pl.DataFrame,
    *,
    scale_type: str | None,
    hand: str | None,
    n: int | None,
    top_n: int,
) -> pl.DataFrame:
    filtered = figure_filter_frame(frame, scale_type=scale_type, hand=hand, n=n)
    return filtered.head(top_n)


def figure_filter_frame(
    frame: pl.DataFrame,
    *,
    scale_type: str | None,
    hand: str | None,
    n: int | None,
) -> pl.DataFrame:
    filtered = frame
    if scale_type is not None:
        filtered = filtered.filter(pl.col(SCALE_TYPE_COLUMN) == scale_type)
    if hand is not None:
        filtered = filtered.filter(pl.col(HAND_COLUMN) == hand)
    if n is not None:
        filtered = filtered.filter(pl.col(N_COLUMN) == n)

    if filtered.is_empty():
        return filtered

    total = int(filtered[COUNT_COLUMN].sum())
    result = filtered.sort(COUNT_COLUMN, descending=True, maintain_order=True)
    result = result.with_columns((pl.col(COUNT_COLUMN) / max(total, 1)).alias(FIGURE_PERCENT_COLUMN))
    result = result.with_columns(pl.Series(FIGURE_LABEL_COLUMN, [f"#{index + 1}" for index in range(result.height)]))
    return _add_figure_annotations(result)


def figure_property_distribution(frame: pl.DataFrame) -> pl.DataFrame:
    schema = {
        FIGURE_PROPERTY_COLUMN: pl.String(),
        FIGURE_PROPERTY_VALUE_COLUMN: pl.Boolean(),
        COUNT_COLUMN: pl.Int64(),
        FIGURE_PERCENT_COLUMN: pl.Float64(),
    }
    if frame.is_empty():
        return pl.DataFrame(schema=schema)

    annotated = _add_figure_annotations(frame)
    total = int(annotated[COUNT_COLUMN].sum())
    rows: list[dict[str, object]] = []
    for property_column in _FIGURE_PROPERTY_COLUMNS:
        for value in (True, False):
            count = int(annotated.filter(pl.col(property_column) == value)[COUNT_COLUMN].sum())
            rows.append(
                {
                    FIGURE_PROPERTY_COLUMN: property_column,
                    FIGURE_PROPERTY_VALUE_COLUMN: value,
                    COUNT_COLUMN: count,
                    FIGURE_PERCENT_COLUMN: count / max(total, 1),
                }
            )

    return pl.DataFrame(rows, schema=schema, orient="row")


def parse_figure_ngram(value: str) -> FigureNGram:
    return FigureNGram.model_validate_json(value)


def _add_figure_annotations(frame: pl.DataFrame) -> pl.DataFrame:
    if all(column in frame.columns for column in (FIGURE_TEXT_COLUMN, *_FIGURE_PROPERTY_COLUMNS)):
        return frame

    figures = [parse_figure_ngram(value) for value in frame[FIGURE_COLUMN].to_list()]
    return frame.with_columns(
        pl.Series(FIGURE_TEXT_COLUMN, [str(figure) for figure in figures]),
        pl.Series(FIGURE_MONOPHONIC_COLUMN, [figure.monophonic for figure in figures]),
        pl.Series(FIGURE_CHORDS_ONLY_COLUMN, [figure.chords_only for figure in figures]),
        pl.Series(FIGURE_IN_SCALE_COLUMN, [figure.in_scale for figure in figures]),
    )


def figure_ngram_to_score_data(
    figure: FigureNGram,
    *,
    preferred_unit: Fraction = _DEFAULT_PREFERRED_UNIT,
    anchor_octave: int = 4,
) -> ScoreData:
    display_unit = figure_display_unit(figure, preferred_unit=preferred_unit)
    notes: list[NoteData] = []
    for onset_degrees, normalized_duration in figure.onsets:
        duration, dots = _vexflow_duration(normalized_duration * display_unit)
        keys: list[str] = []
        accidentals: list[VexflowAccidental] = []
        for relative_position, accidental in onset_degrees:
            keys.append(_relative_degree_key(relative_position, anchor_octave=anchor_octave))
            accidentals.append(_ACCIDENTAL_BY_OFFSET[accidental])

        notes.append(NoteData(keys=keys, accidentals=accidentals, duration=duration, dots=dots))

    return ScoreData(
        rows=[
            [
                StaveData(
                    clef="treble",
                    key_signature="C",
                    time_signature=None,
                    voices=[VoiceData(notes=notes)],
                )
            ]
        ],
        max_notes_per_measure=len(notes),
    )


def figure_display_unit(
    figure: FigureNGram,
    *,
    preferred_unit: Fraction = _DEFAULT_PREFERRED_UNIT,
) -> Fraction:
    candidates = tuple(dict.fromkeys((preferred_unit, *_DISPLAY_UNIT_CANDIDATES)))
    normalized_durations = tuple(duration for _, duration in figure.onsets)
    for unit in candidates:
        if all(duration * unit in _REPRESENTABLE_DURATIONS for duration in normalized_durations):
            return unit

    raise ValueError("figure durations cannot be represented with available VexFlow note values")


def _require_figure_count_columns(frame: pl.DataFrame) -> None:
    missing = _FIGURE_COUNT_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing required figure count columns: {missing_text}")


def _relative_degree_key(relative_position: int, *, anchor_octave: int) -> str:
    octave_offset, letter_index = divmod(relative_position, len(_LETTER_NAMES))
    return f"{_LETTER_NAMES[letter_index]}/{anchor_octave + octave_offset}"


def _vexflow_duration(duration: Fraction) -> tuple[VexflowDuration, int]:
    try:
        return _REPRESENTABLE_DURATIONS[duration]
    except KeyError as exception:
        raise ValueError(f"unsupported VexFlow duration: {duration}") from exception
