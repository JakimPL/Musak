from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from musak_model.processing.manifest import (
    EncodedManifestField,
    ParsedManifestField,
    ParsedManifestStatus,
)
from musak_model.processing.paths import ENCODED_MANIFEST_NAME, PARSED_MANIFEST_NAME
from musak_shared.elements import KEYS

COUNT_COLUMN: Final[str] = "count"
PERCENT_COLUMN: Final[str] = "percent"
VALUE_COLUMN: Final[str] = "value"
LABEL_COLUMN: Final[str] = "Metric"
METRIC_VALUE_COLUMN: Final[str] = "Value"
TOKEN_BIN_START_COLUMN: Final[str] = "token_bin_start"
TOKEN_BIN_END_COLUMN: Final[str] = "token_bin_end"
TOKEN_BIN_LABEL_COLUMN: Final[str] = "token_bin"
TOKEN_BIN_MIDPOINT_COLUMN: Final[str] = "token_bin_midpoint"
TOKEN_BAR_START_COLUMN: Final[str] = "token_bar_start"
TOKEN_BAR_END_COLUMN: Final[str] = "token_bar_end"

_TRUE_TEXT: Final[str] = "True"
_REASON_SEPARATOR: Final[str] = "|"
_DIAGNOSTIC_NUMERIC_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.RIGHT_SILENCE_FRACTION,
    EncodedManifestField.LEFT_SILENCE_FRACTION,
    EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION,
    EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION,
    EncodedManifestField.RIGHT_ONLY_ACTIVE_FRACTION,
    EncodedManifestField.LEFT_ONLY_ACTIVE_FRACTION,
    EncodedManifestField.LONGEST_RIGHT_SILENCE_BEATS,
    EncodedManifestField.LONGEST_LEFT_SILENCE_BEATS,
    EncodedManifestField.LONGEST_BOTH_HANDS_SILENCE_BEATS,
    EncodedManifestField.RIGHT_NOTE_ONSETS_PER_BAR,
    EncodedManifestField.LEFT_NOTE_ONSETS_PER_BAR,
    EncodedManifestField.SILENT_BAR_COUNT,
    EncodedManifestField.SILENT_BAR_FRACTION,
    EncodedManifestField.SILENT_EDGE_BAR_COUNT,
    EncodedManifestField.HAND_ACTIVITY_BALANCE,
    EncodedManifestField.NOTE_TOKEN_FRACTION,
    EncodedManifestField.REST_TOKEN_FRACTION,
    EncodedManifestField.HOLD_TOKEN_FRACTION,
    EncodedManifestField.SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_EXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_UNEXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_BEST_MARGIN,
    EncodedManifestField.SCALE_MATCH_OBSERVED_PITCH_CLASS_COUNT,
    EncodedManifestField.SCALE_MATCH_EXPLANATION_PITCH_CLASS_COUNT,
    EncodedManifestField.SCALE_MATCH_SUPPORT_CANDIDATE_COUNT,
    EncodedManifestField.SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT,
)
_DIAGNOSTIC_BOOLEAN_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.EMPTY_SCORE,
    EncodedManifestField.ONE_HAND_ONLY,
    EncodedManifestField.SCALE_MATCH_DECLARED_MATCH_USED,
    EncodedManifestField.SCALE_MATCH_LOW_CONFIDENCE,
    EncodedManifestField.SCALE_MATCH_AMBIGUOUS,
    EncodedManifestField.SCALE_MATCH_NO_PITCHES,
)


@dataclass(frozen=True)
class DatasetStatistics:
    parsed: pd.DataFrame
    encoded: pd.DataFrame | None

    @property
    def has_encoded(self) -> bool:
        return self.encoded is not None


def processed_dataset_dirs(processed_root: Path) -> list[Path]:
    if not processed_root.exists():
        return []

    return sorted(path for path in processed_root.iterdir() if path.is_dir())


def encoded_run_dirs(dataset_dir: Path) -> list[Path]:
    encoded_root = dataset_dir / "encoded"
    if not encoded_root.exists():
        return []

    return sorted(path for path in encoded_root.iterdir() if (path / ENCODED_MANIFEST_NAME).is_file())


def load_dataset_statistics(dataset_dir: Path, encoded_dir: Path | None) -> DatasetStatistics:
    parsed = read_parsed_manifest_frame(dataset_dir / PARSED_MANIFEST_NAME)
    encoded = read_encoded_manifest_frame(encoded_dir / ENCODED_MANIFEST_NAME) if encoded_dir is not None else None
    return DatasetStatistics(parsed=parsed, encoded=encoded)


def read_parsed_manifest_frame(path: Path) -> pd.DataFrame:
    frame = _read_manifest_frame(path)
    _require_columns(frame, ParsedManifestField)
    frame[ParsedManifestField.RIGHT_HAND_BARS] = _numeric_series(frame[ParsedManifestField.RIGHT_HAND_BARS])
    frame[ParsedManifestField.LEFT_HAND_BARS] = _numeric_series(frame[ParsedManifestField.LEFT_HAND_BARS])
    frame[ParsedManifestField.DECLARED_KEY_FIFTHS] = _numeric_series(frame[ParsedManifestField.DECLARED_KEY_FIFTHS])
    frame["has_parse_diagnostics"] = frame[ParsedManifestField.PARSE_DIAGNOSTICS] != ""
    return frame


def read_encoded_manifest_frame(path: Path) -> pd.DataFrame:
    frame = _read_manifest_frame(path)
    _require_columns(frame, EncodedManifestField)
    frame[EncodedManifestField.ELIGIBLE_FOR_TRAINING] = frame[EncodedManifestField.ELIGIBLE_FOR_TRAINING] == _TRUE_TEXT
    frame[EncodedManifestField.ENCODED_LINE] = _numeric_series(frame[EncodedManifestField.ENCODED_LINE])
    frame[EncodedManifestField.WINDOW_START_BAR] = _numeric_series(frame[EncodedManifestField.WINDOW_START_BAR])
    frame[EncodedManifestField.BAR_COUNT] = _numeric_series(frame[EncodedManifestField.BAR_COUNT])
    frame[EncodedManifestField.TOKEN_COUNT] = _numeric_series(frame[EncodedManifestField.TOKEN_COUNT])
    frame[EncodedManifestField.DIFFICULTY_LEVEL] = _numeric_series(frame[EncodedManifestField.DIFFICULTY_LEVEL])
    for field in _DIAGNOSTIC_NUMERIC_FIELDS:
        frame[field] = _numeric_series(frame[field])
    for field in _DIAGNOSTIC_BOOLEAN_FIELDS:
        frame[field] = frame[field] == _TRUE_TEXT
    return frame


def overview_rows(stats: DatasetStatistics) -> list[dict[str, str]]:
    parsed = stats.parsed
    parsed_count = len(parsed)
    parsed_successes = int((parsed[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value).sum())
    parsed_errors = parsed_count - parsed_successes
    rows = [
        _metric_row("Parsed files", parsed_count),
        _metric_row("Parsed successfully", parsed_successes),
        _metric_row("Parse errors", parsed_errors),
        _metric_row("Parse success rate", _percent(parsed_successes, parsed_count)),
    ]

    if stats.encoded is None:
        return rows

    encoded = stats.encoded
    segment_count = len(encoded)
    eligible_count = int(encoded[EncodedManifestField.ELIGIBLE_FOR_TRAINING].sum())
    encoded_count = int(encoded[EncodedManifestField.ENCODED_LINE].notna().sum())
    rows.extend(
        [
            _metric_row("Segments", segment_count),
            _metric_row("Eligible segments", eligible_count),
            _metric_row("Eligibility rate", _percent(eligible_count, segment_count)),
            _metric_row("Encoded samples", encoded_count),
        ]
    )
    return rows


def categorical_distribution(
    frame: pd.DataFrame,
    column: str,
    *,
    top_n: int,
    empty_label: str = "(empty)",
) -> pd.DataFrame:
    values = frame[column].astype("string").fillna("").replace("", empty_label)
    counts = values.value_counts(dropna=False).head(top_n).rename_axis(VALUE_COLUMN).reset_index(name=COUNT_COLUMN)
    counts[PERCENT_COLUMN] = counts[COUNT_COLUMN] / max(len(frame), 1)
    return counts


def scale_root_distribution(frame: pd.DataFrame, column: str, *, top_n: int) -> pd.DataFrame:
    distribution = categorical_distribution(frame, column, top_n=top_n)
    distribution[VALUE_COLUMN] = distribution[VALUE_COLUMN].map(_scale_root_label)
    return distribution


def eligibility_distribution(encoded: pd.DataFrame) -> pd.DataFrame:
    labels = encoded[EncodedManifestField.ELIGIBLE_FOR_TRAINING].map({True: "eligible", False: "ineligible"})
    counts = labels.value_counts(dropna=False).rename_axis(VALUE_COLUMN).reset_index(name=COUNT_COLUMN)
    counts[PERCENT_COLUMN] = counts[COUNT_COLUMN] / max(len(encoded), 1)
    return counts


def ineligibility_reason_distribution(encoded: pd.DataFrame) -> pd.DataFrame:
    reasons = _explode_reasons(encoded)
    if reasons.empty:
        return pd.DataFrame(columns=[VALUE_COLUMN, COUNT_COLUMN, PERCENT_COLUMN])

    counts = reasons[VALUE_COLUMN].value_counts().rename_axis(VALUE_COLUMN).reset_index(name=COUNT_COLUMN)
    counts[PERCENT_COLUMN] = counts[COUNT_COLUMN] / max(len(encoded), 1)
    return counts


def reason_by_column(encoded: pd.DataFrame, column: str) -> pd.DataFrame:
    column_name = str(column)
    reasons = _explode_reasons(encoded)
    if reasons.empty:
        return pd.DataFrame(columns=[VALUE_COLUMN, column_name, COUNT_COLUMN])

    return reasons.groupby([VALUE_COLUMN, column_name], dropna=False).size().reset_index(name=COUNT_COLUMN)


def table_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records = frame.to_dict("records")
    return [{str(key): value for key, value in row.items()} for row in records]


def selected_table_row(table: object) -> dict[str, object] | None:
    value = getattr(table, "value", None)
    if value is None:
        return None

    if isinstance(value, pd.DataFrame):
        records = table_records(value)
        return records[0] if records else None

    if isinstance(value, list):
        if not value:
            return None

        first = value[0]
        return {str(key): item for key, item in first.items()} if isinstance(first, dict) else None

    if isinstance(value, dict):
        rows = value.get("rows")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return {str(key): item for key, item in rows[0].items()}

        return {str(key): item for key, item in value.items()}

    return None


def token_summary_rows(encoded: pd.DataFrame) -> list[dict[str, str]]:
    tokens = encoded[EncodedManifestField.TOKEN_COUNT].dropna()
    if tokens.empty:
        return [_metric_row(metric, "") for metric in ("min", "p25", "median", "mean", "p75", "p95", "max")]

    return [
        _metric_row("min", int(tokens.min())),
        _metric_row("p25", int(tokens.quantile(0.25))),
        _metric_row("median", int(tokens.median())),
        _metric_row("mean", f"{tokens.mean():.1f}"),
        _metric_row("p75", int(tokens.quantile(0.75))),
        _metric_row("p95", int(tokens.quantile(0.95))),
        _metric_row("max", int(tokens.max())),
    ]


def diagnostic_summary_rows(encoded: pd.DataFrame) -> list[dict[str, str]]:
    rows = [
        _metric_row(
            "empty score rate",
            _percent(int(encoded[EncodedManifestField.EMPTY_SCORE].sum()), len(encoded)),
        ),
        _metric_row(
            "one hand only rate",
            _percent(int(encoded[EncodedManifestField.ONE_HAND_ONLY].sum()), len(encoded)),
        ),
    ]
    rows.extend(
        [
            _mean_metric_row("right silence", encoded[EncodedManifestField.RIGHT_SILENCE_FRACTION], percent=True),
            _mean_metric_row("left silence", encoded[EncodedManifestField.LEFT_SILENCE_FRACTION], percent=True),
            _mean_metric_row(
                "both hands silence",
                encoded[EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION],
                percent=True,
            ),
            _mean_metric_row(
                "both hands active",
                encoded[EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION],
                percent=True,
            ),
            _mean_metric_row("hand activity balance", encoded[EncodedManifestField.HAND_ACTIVITY_BALANCE]),
            _mean_metric_row("silent bars", encoded[EncodedManifestField.SILENT_BAR_COUNT]),
            _mean_metric_row("silent bar share", encoded[EncodedManifestField.SILENT_BAR_FRACTION], percent=True),
            _mean_metric_row("silent edge bars", encoded[EncodedManifestField.SILENT_EDGE_BAR_COUNT]),
            _mean_metric_row("note token share", encoded[EncodedManifestField.NOTE_TOKEN_FRACTION], percent=True),
            _mean_metric_row("rest token share", encoded[EncodedManifestField.REST_TOKEN_FRACTION], percent=True),
            _mean_metric_row(
                "in-scale pitch duration",
                encoded[EncodedManifestField.SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION],
                percent=True,
            ),
            _mean_metric_row("scale match margin", encoded[EncodedManifestField.SCALE_MATCH_BEST_MARGIN]),
        ]
    )
    return rows


def diagnostic_bucket_distribution(
    encoded: pd.DataFrame,
    column: str,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    values = encoded[column].dropna().clip(lower=0, upper=1)
    if values.empty:
        return pd.DataFrame(columns=[VALUE_COLUMN, COUNT_COLUMN, PERCENT_COLUMN])

    bin_edges = [index / bins for index in range(bins + 1)]
    buckets = pd.cut(values, bins=bin_edges, include_lowest=True, right=True)
    counts = buckets.value_counts(sort=False).rename_axis("_bucket").reset_index(name=COUNT_COLUMN)
    counts[VALUE_COLUMN] = counts["_bucket"].map(lambda interval: f"{interval.left:.1f}-{interval.right:.1f}")
    counts[PERCENT_COLUMN] = counts[COUNT_COLUMN] / max(len(values), 1)
    return counts[[VALUE_COLUMN, COUNT_COLUMN, PERCENT_COLUMN]]


def token_histogram_distribution(encoded: pd.DataFrame, *, bins: int = 40) -> pd.DataFrame:
    tokens = encoded[EncodedManifestField.TOKEN_COUNT].dropna()
    if tokens.empty:
        return pd.DataFrame(
            columns=[
                TOKEN_BIN_START_COLUMN,
                TOKEN_BIN_END_COLUMN,
                TOKEN_BIN_LABEL_COLUMN,
                TOKEN_BIN_MIDPOINT_COLUMN,
                TOKEN_BAR_START_COLUMN,
                TOKEN_BAR_END_COLUMN,
                EncodedManifestField.ELIGIBLE_FOR_TRAINING,
                COUNT_COLUMN,
            ]
        )

    bin_width = _nice_bin_width(float(tokens.max()), bins=bins)
    max_edge = (math.floor(float(tokens.max()) / bin_width) + 1) * bin_width
    bin_edges = list(range(0, int(max_edge + bin_width), int(bin_width)))
    binned = encoded.loc[
        tokens.index, [EncodedManifestField.TOKEN_COUNT, EncodedManifestField.ELIGIBLE_FOR_TRAINING]
    ].copy()
    binned["_token_interval"] = pd.cut(
        binned[EncodedManifestField.TOKEN_COUNT],
        bins=bin_edges,
        right=False,
        include_lowest=True,
    )
    counts = (
        binned.groupby(["_token_interval", EncodedManifestField.ELIGIBLE_FOR_TRAINING], observed=True)
        .size()
        .reset_index(name=COUNT_COLUMN)
    )
    counts[TOKEN_BIN_START_COLUMN] = counts["_token_interval"].map(lambda interval: float(interval.left))
    counts[TOKEN_BIN_END_COLUMN] = counts["_token_interval"].map(lambda interval: float(interval.right))
    counts[TOKEN_BIN_MIDPOINT_COLUMN] = (
        counts[TOKEN_BIN_START_COLUMN].astype(float) + counts[TOKEN_BIN_END_COLUMN].astype(float)
    ) / 2
    bar_padding = bin_width * 0.04
    counts[TOKEN_BAR_START_COLUMN] = counts[TOKEN_BIN_START_COLUMN].astype(float) + bar_padding
    counts[TOKEN_BAR_END_COLUMN] = counts[TOKEN_BIN_END_COLUMN].astype(float) - bar_padding
    counts[TOKEN_BIN_LABEL_COLUMN] = counts["_token_interval"].map(
        lambda interval: f"{interval.left:.0f}-{interval.right:.0f}"
    )
    return counts.drop(columns=["_token_interval"]).sort_values(
        [TOKEN_BIN_START_COLUMN, COUNT_COLUMN],
        ascending=[True, False],
    )


def parse_error_table_frame(parsed: pd.DataFrame) -> pd.DataFrame:
    errors = parsed[parsed[ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value]
    columns = [
        ParsedManifestField.SOURCE_PATH,
        ParsedManifestField.ERROR_TYPE,
        ParsedManifestField.ERROR_MESSAGE,
        ParsedManifestField.PARSE_DIAGNOSTICS,
    ]
    return _table_frame(errors, columns)


def parsed_table_frame(parsed: pd.DataFrame) -> pd.DataFrame:
    columns = [
        ParsedManifestField.SOURCE_PATH,
        ParsedManifestField.STATUS,
        ParsedManifestField.ERROR_TYPE,
        ParsedManifestField.DECLARED_KEY_FIFTHS,
        ParsedManifestField.TIME_SIGNATURE,
        ParsedManifestField.RIGHT_HAND_BARS,
        ParsedManifestField.LEFT_HAND_BARS,
    ]
    return _table_frame(parsed, columns)


def encoded_table_frame(encoded: pd.DataFrame) -> pd.DataFrame:
    columns = [
        EncodedManifestField.SEGMENT_ID,
        EncodedManifestField.SOURCE_ID,
        EncodedManifestField.SOURCE_PATH,
        EncodedManifestField.PARSED_PATH,
        EncodedManifestField.ENCODED_SHARD,
        EncodedManifestField.ENCODED_LINE,
        EncodedManifestField.WINDOW_START_BAR,
        EncodedManifestField.BAR_COUNT,
        EncodedManifestField.TOKEN_COUNT,
        EncodedManifestField.ELIGIBLE_FOR_TRAINING,
        EncodedManifestField.INELIGIBILITY_REASONS,
        EncodedManifestField.SCALE_ROOT,
        EncodedManifestField.SCALE_TYPE,
        EncodedManifestField.DECLARED_KEY_FIFTHS,
        EncodedManifestField.SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION,
        EncodedManifestField.SCALE_MATCH_UNEXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION,
        EncodedManifestField.SCALE_MATCH_BEST_MARGIN,
        EncodedManifestField.SCALE_MATCH_EXPLANATION_PITCH_CLASS_COUNT,
        EncodedManifestField.SCALE_MATCH_LOW_CONFIDENCE,
        EncodedManifestField.SCALE_MATCH_AMBIGUOUS,
        EncodedManifestField.TIME_SIGNATURE,
        EncodedManifestField.DIFFICULTY_LEVEL,
        EncodedManifestField.RIGHT_SILENCE_FRACTION,
        EncodedManifestField.LEFT_SILENCE_FRACTION,
        EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION,
        EncodedManifestField.HAND_ACTIVITY_BALANCE,
        EncodedManifestField.EMPTY_SCORE,
        EncodedManifestField.ONE_HAND_ONLY,
    ]
    return _table_frame(encoded, columns)


def _explode_reasons(encoded: pd.DataFrame) -> pd.DataFrame:
    reason_rows: list[dict[str, object]] = []
    for _, row in encoded.iterrows():
        reason_text = row[EncodedManifestField.INELIGIBILITY_REASONS]
        if not isinstance(reason_text, str) or reason_text == "":
            continue

        for reason in reason_text.split(_REASON_SEPARATOR):
            reason_rows.append(
                {
                    VALUE_COLUMN: reason,
                    EncodedManifestField.TIME_SIGNATURE: row[EncodedManifestField.TIME_SIGNATURE],
                    EncodedManifestField.SCALE_ROOT: row[EncodedManifestField.SCALE_ROOT],
                    EncodedManifestField.SCALE_TYPE: row[EncodedManifestField.SCALE_TYPE],
                }
            )

    return pd.DataFrame(reason_rows)


def _read_manifest_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _require_columns(frame: pd.DataFrame, field_type: type[EncodedManifestField] | type[ParsedManifestField]) -> None:
    missing = [field.value for field in field_type if field.value not in frame.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"manifest is missing required columns: {missing_text}")


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", pd.NA), errors="coerce")


def _metric_row(label: str, value: object) -> dict[str, str]:
    return {LABEL_COLUMN: label, METRIC_VALUE_COLUMN: str(value)}


def _mean_metric_row(label: str, values: pd.Series, *, percent: bool = False) -> dict[str, str]:
    clean_values = values.dropna()
    if clean_values.empty:
        return _metric_row(label, "")

    mean_value = float(clean_values.mean())
    if percent:
        return _metric_row(label, f"{100 * mean_value:.1f}%")

    return _metric_row(label, f"{mean_value:.3f}")


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"

    return f"{100 * numerator / denominator:.1f}%"


def _table_frame(frame: pd.DataFrame, columns: Sequence[object]) -> pd.DataFrame:
    table = frame.loc[:, [str(column) for column in columns]].copy()
    table.columns = [str(column) for column in table.columns]
    return table


def _scale_root_label(value: object) -> str:
    try:
        if not isinstance(value, (int, float, str)):
            return str(value)

        return KEYS[int(value)]
    except (TypeError, ValueError, KeyError):
        return str(value)


def _nice_bin_width(max_value: float, *, bins: int) -> int:
    if max_value <= 0:
        return 1

    raw_width = max_value / max(bins, 1)
    magnitude = 10 ** math.floor(math.log10(raw_width))
    for multiplier in (1, 2, 5, 10):
        width = multiplier * magnitude
        if width >= raw_width:
            return max(1, int(width))

    return max(1, int(10 * magnitude))
