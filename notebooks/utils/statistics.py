from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from musak_model.processing.manifest import (
    EncodedManifestField,
    ParsedManifestField,
    ParsedManifestStatus,
)

COUNT_COLUMN: Final[str] = "count"
PERCENT_COLUMN: Final[str] = "percent"
VALUE_COLUMN: Final[str] = "value"
LABEL_COLUMN: Final[str] = "Metric"
METRIC_VALUE_COLUMN: Final[str] = "Value"

_TRUE_TEXT: Final[str] = "True"
_REASON_SEPARATOR: Final[str] = "|"


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

    return sorted(path for path in encoded_root.iterdir() if (path / "encoded.csv").is_file())


def load_dataset_statistics(dataset_dir: Path, encoded_dir: Path | None) -> DatasetStatistics:
    parsed = read_parsed_manifest_frame(dataset_dir / "parsed.csv")
    encoded = read_encoded_manifest_frame(encoded_dir / "encoded.csv") if encoded_dir is not None else None
    return DatasetStatistics(parsed=parsed, encoded=encoded)


def read_parsed_manifest_frame(path: Path) -> pd.DataFrame:
    frame = _read_manifest_frame(path)
    _require_columns(frame, ParsedManifestField)
    frame[ParsedManifestField.RIGHT_HAND_BARS] = _numeric_series(frame[ParsedManifestField.RIGHT_HAND_BARS])
    frame[ParsedManifestField.LEFT_HAND_BARS] = _numeric_series(frame[ParsedManifestField.LEFT_HAND_BARS])
    frame[ParsedManifestField.KEY_ROOT] = _numeric_series(frame[ParsedManifestField.KEY_ROOT])
    frame[ParsedManifestField.KEY_FIFTHS] = _numeric_series(frame[ParsedManifestField.KEY_FIFTHS])
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
    reasons = _explode_reasons(encoded)
    if reasons.empty:
        return pd.DataFrame(columns=[VALUE_COLUMN, column, COUNT_COLUMN])

    return reasons.groupby([VALUE_COLUMN, column], dropna=False).size().reset_index(name=COUNT_COLUMN)


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


def top_parse_error_rows(parsed: pd.DataFrame, *, limit: int) -> list[dict[str, str]]:
    errors = parsed[parsed[ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value]
    columns = [
        ParsedManifestField.SOURCE_PATH,
        ParsedManifestField.ERROR_TYPE,
        ParsedManifestField.ERROR_MESSAGE,
        ParsedManifestField.PARSE_DIAGNOSTICS,
    ]
    return errors.loc[:, columns].head(limit).to_dict("records")


def parsed_table_rows(parsed: pd.DataFrame, *, limit: int) -> list[dict[str, object]]:
    columns = [
        ParsedManifestField.SOURCE_PATH,
        ParsedManifestField.STATUS,
        ParsedManifestField.ERROR_TYPE,
        ParsedManifestField.KEY_ROOT,
        ParsedManifestField.SCALE_TYPE,
        ParsedManifestField.TIME_SIGNATURE,
        ParsedManifestField.RIGHT_HAND_BARS,
        ParsedManifestField.LEFT_HAND_BARS,
    ]
    return parsed.loc[:, columns].head(limit).to_dict("records")


def encoded_table_rows(encoded: pd.DataFrame, *, limit: int) -> list[dict[str, object]]:
    columns = [
        EncodedManifestField.SEGMENT_ID,
        EncodedManifestField.SOURCE_PATH,
        EncodedManifestField.WINDOW_START_BAR,
        EncodedManifestField.BAR_COUNT,
        EncodedManifestField.TOKEN_COUNT,
        EncodedManifestField.ELIGIBLE_FOR_TRAINING,
        EncodedManifestField.INELIGIBILITY_REASONS,
        EncodedManifestField.KEY_ROOT,
        EncodedManifestField.SCALE_TYPE,
        EncodedManifestField.TIME_SIGNATURE,
        EncodedManifestField.DIFFICULTY_LEVEL,
    ]
    return encoded.loc[:, columns].head(limit).to_dict("records")


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
                    EncodedManifestField.KEY_ROOT: row[EncodedManifestField.KEY_ROOT],
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


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"

    return f"{100 * numerator / denominator:.1f}%"
