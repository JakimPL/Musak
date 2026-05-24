from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Final

import pandas as pd

from musak_model.analysis.n_grams import FigureNGram
from musak_model.analysis.n_grams.export import (
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

FIGURE_PERCENT_COLUMN: Final[str] = "percent"
FIGURE_TOTAL_COLUMN: Final[str] = "total_count"
FIGURE_UNIQUE_COLUMN: Final[str] = "unique_figures"
FIGURE_DURATION_UNIT_COLUMN: Final[str] = "duration_unit"
FIGURE_LABEL_COLUMN: Final[str] = "label"
FIGURE_TEXT_COLUMN: Final[str] = "figure_text"

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


def analysis_result_files(analysis_dir: Path = DEFAULT_ANALYSIS_DIR) -> list[Path]:
    if not analysis_dir.exists():
        return []

    return sorted(path for path in analysis_dir.glob("*.csv") if path.is_file())


def read_figure_count_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_figure_count_columns(frame)
    frame[N_COLUMN] = pd.to_numeric(frame[N_COLUMN], errors="raise").astype(int)
    frame[COUNT_COLUMN] = pd.to_numeric(frame[COUNT_COLUMN], errors="raise").astype(int)
    return frame


def figure_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[SCALE_TYPE_COLUMN, HAND_COLUMN, N_COLUMN, FIGURE_TOTAL_COLUMN, FIGURE_UNIQUE_COLUMN]
        )

    return (
        frame.groupby([SCALE_TYPE_COLUMN, HAND_COLUMN, N_COLUMN], dropna=False)
        .agg(
            **{
                FIGURE_TOTAL_COLUMN: (COUNT_COLUMN, "sum"),
                FIGURE_UNIQUE_COLUMN: (FIGURE_COLUMN, "count"),
            }
        )
        .reset_index()
        .sort_values([SCALE_TYPE_COLUMN, HAND_COLUMN, N_COLUMN])
    )


def top_figure_frame(
    frame: pd.DataFrame,
    *,
    scale_type: str | None,
    hand: str | None,
    n: int | None,
    top_n: int,
) -> pd.DataFrame:
    filtered = figure_filter_frame(frame, scale_type=scale_type, hand=hand, n=n)
    return filtered.head(top_n)


def figure_filter_frame(
    frame: pd.DataFrame,
    *,
    scale_type: str | None,
    hand: str | None,
    n: int | None,
) -> pd.DataFrame:
    filtered = frame
    if scale_type is not None:
        filtered = filtered[filtered[SCALE_TYPE_COLUMN] == scale_type]
    if hand is not None:
        filtered = filtered[filtered[HAND_COLUMN] == hand]
    if n is not None:
        filtered = filtered[filtered[N_COLUMN] == n]

    if filtered.empty:
        return filtered.copy()

    total = int(filtered[COUNT_COLUMN].sum())
    result = filtered.sort_values(COUNT_COLUMN, ascending=False).copy()
    result[FIGURE_PERCENT_COLUMN] = result[COUNT_COLUMN] / max(total, 1)
    result[FIGURE_LABEL_COLUMN] = [f"#{index + 1}" for index in range(len(result))]
    result[FIGURE_TEXT_COLUMN] = [str(parse_figure_ngram(str(value))) for value in result[FIGURE_COLUMN]]
    return result


def parse_figure_ngram(value: str) -> FigureNGram:
    return FigureNGram.model_validate_json(value)


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


def _require_figure_count_columns(frame: pd.DataFrame) -> None:
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
