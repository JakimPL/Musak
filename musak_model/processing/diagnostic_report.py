from __future__ import annotations

import csv
import heapq
import json
import logging
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

from musak_model.data.converter import pitch_to_degree
from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.data.scale_matcher.matcher import match_scale_histogram
from musak_model.processing.io import load_tokenizer_snapshot_json
from musak_model.processing.manifest import EncodedManifestField, ParsedManifestField, ParsedManifestStatus
from musak_model.processing.paths import ENCODED_JSONL_NAME, ENCODED_MANIFEST_NAME, PARSED_MANIFEST_NAME
from musak_model.processing.snapshot import TokenizerSnapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    StartToken,
)
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_shared.elements import KEYS
from musak_shared.files import JSON_INDENT, write_csv_rows

_LOGGER = logging.getLogger(__name__)

REPORT_NAME: Final[str] = "report.md"
SUMMARY_NAME: Final[str] = "summary.json"
TABLES_DIRECTORY_NAME: Final[str] = "tables"
DEFAULT_TOP_ROWS: Final[int] = 25
DEFAULT_RARE_TOKEN_THRESHOLD: Final[int] = 5
DEFAULT_TOKEN_ID_TABLE_LIMIT: Final[int] = 100
_REASON_SEPARATOR: Final[str] = "|"
_A_NATURAL_MINOR_PITCH_CLASSES: Final[tuple[int, ...]] = (9, 11, 0, 2, 4, 5, 7)
_A_HARMONIC_MINOR_PITCH_CLASSES: Final[tuple[int, ...]] = (9, 11, 0, 2, 4, 5, 8)
_A_MELODIC_MINOR_PITCH_CLASSES: Final[tuple[int, ...]] = (9, 11, 0, 2, 4, 6, 8)
_A4_MIDI_PITCH: Final[int] = 69
_DECLARED_C_MAJOR_FIFTHS: Final[int] = 0
_COUNT_COLUMN: Final[str] = "count"
_PERCENT_COLUMN: Final[str] = "percent"
_VALUE_COLUMN: Final[str] = "value"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
CsvRow = dict[str, JsonScalar]

_NUMERIC_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.TOKEN_COUNT,
    EncodedManifestField.SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_BEST_MARGIN,
    EncodedManifestField.SCALE_MATCH_SUPPORT_CANDIDATE_COUNT,
    EncodedManifestField.SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT,
    EncodedManifestField.ACCIDENTAL_NOTE_FRACTION,
    EncodedManifestField.IN_SCALE_NOTE_FRACTION,
    EncodedManifestField.NOTE_DENSITY_PER_BEAT,
    EncodedManifestField.ONSET_DENSITY_PER_BEAT,
    EncodedManifestField.RIGHT_ONSET_DENSITY_PER_BEAT,
    EncodedManifestField.LEFT_ONSET_DENSITY_PER_BEAT,
    EncodedManifestField.SHORTEST_NOTE_DURATION_BEATS,
    EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION,
    EncodedManifestField.HAND_ACTIVITY_BALANCE,
    EncodedManifestField.MAX_NOTES_PER_ONSET,
    EncodedManifestField.MAX_NOTES_PER_HAND,
    EncodedManifestField.MAX_ONSET_SPAN_SEMITONES,
    EncodedManifestField.MAX_MELODIC_GAP_SEMITONES,
    EncodedManifestField.STATIC_HAND_SPAN_DEGREES,
    EncodedManifestField.SYNCHRONIZED_ONSET_FRACTION,
    EncodedManifestField.INDEPENDENT_ONSET_FRACTION,
)
_BOOLEAN_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.EMPTY_SCORE,
    EncodedManifestField.ONE_HAND_ONLY,
    EncodedManifestField.SCALE_MATCH_DECLARED_MATCH_USED,
    EncodedManifestField.SCALE_MATCH_LOW_CONFIDENCE,
    EncodedManifestField.SCALE_MATCH_AMBIGUOUS,
    EncodedManifestField.SCALE_MATCH_NO_PITCHES,
    EncodedManifestField.HAS_DOTTED_NOTES,
)
_REFERENCE_NUMERIC_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.TOKEN_COUNT,
    EncodedManifestField.ACCIDENTAL_NOTE_FRACTION,
    EncodedManifestField.IN_SCALE_NOTE_FRACTION,
    EncodedManifestField.NOTE_DENSITY_PER_BEAT,
    EncodedManifestField.ONSET_DENSITY_PER_BEAT,
    EncodedManifestField.MAX_NOTES_PER_ONSET,
    EncodedManifestField.MAX_NOTES_PER_HAND,
    EncodedManifestField.MAX_MELODIC_GAP_SEMITONES,
    EncodedManifestField.STATIC_HAND_SPAN_DEGREES,
    EncodedManifestField.SCALE_MATCH_BEST_MARGIN,
)
_OUTLIER_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.SEGMENT_ID,
    EncodedManifestField.SOURCE_PATH,
    EncodedManifestField.WINDOW_START_BAR,
    EncodedManifestField.BAR_COUNT,
    EncodedManifestField.TOKEN_COUNT,
    EncodedManifestField.SCALE_ROOT,
    EncodedManifestField.SCALE_TYPE,
    EncodedManifestField.DECLARED_KEY_FIFTHS,
    EncodedManifestField.SPELLING_KEY_FIFTHS,
    EncodedManifestField.SPELLING_CONTEXT_SOURCE,
    EncodedManifestField.SCALE_MATCH_BEST_MARGIN,
    EncodedManifestField.SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT,
    EncodedManifestField.SCALE_MATCH_AMBIGUOUS,
    EncodedManifestField.ACCIDENTAL_NOTE_FRACTION,
    EncodedManifestField.IN_SCALE_NOTE_FRACTION,
    EncodedManifestField.NOTE_DENSITY_PER_BEAT,
    EncodedManifestField.MAX_NOTES_PER_ONSET,
    EncodedManifestField.STATIC_HAND_SPAN_DEGREES,
)


@dataclass(frozen=True)
class DiagnosticReportResult:
    output_directory: Path
    report_path: Path
    summary_path: Path
    table_paths: dict[str, Path]


@dataclass(frozen=True)
class DatasetPaths:
    dataset_name: str
    processed_directory: Path
    encoded_directory: Path
    parsed_manifest_path: Path
    encoded_manifest_path: Path
    encoded_jsonl_path: Path
    tokenizer_snapshot_path: Path


@dataclass(frozen=True)
class CollectedDiagnostics:
    paths: DatasetPaths
    summary: dict[str, JsonValue]
    tables: dict[str, list[CsvRow]]


@dataclass(order=True)
class _ScoredRow:
    score: float
    order: int
    row: CsvRow = field(compare=False)


class _TopRows:
    def __init__(self, *, limit: int) -> None:
        self._limit = limit
        self._rows: list[_ScoredRow] = []
        self._order = 0

    def add(self, *, score: float | None, row: CsvRow) -> None:
        if score is None or self._limit <= 0:
            return

        item = _ScoredRow(score=score, order=self._order, row=row)
        self._order += 1
        if len(self._rows) < self._limit:
            heapq.heappush(self._rows, item)
            return

        heapq.heappushpop(self._rows, item)

    def rows(self) -> list[CsvRow]:
        return [item.row for item in sorted(self._rows, reverse=True)]


def write_dataset_diagnostic_report(
    *,
    dataset_name: str,
    processed_directory: Path,
    encoded_directory: Path,
    output_directory: Path,
    scale_matcher_config: ScaleMatcherConfig,
    reference_dataset_name: str | None,
    reference_processed_directory: Path | None,
    reference_encoded_directory: Path | None,
    max_sequence_length: int,
    top_rows: int = DEFAULT_TOP_ROWS,
    rare_token_threshold: int = DEFAULT_RARE_TOKEN_THRESHOLD,
    mlflow_db_path: Path | None = None,
) -> DiagnosticReportResult:
    primary = collect_dataset_diagnostics(
        dataset_name=dataset_name,
        processed_directory=processed_directory,
        encoded_directory=encoded_directory,
        scale_matcher_config=scale_matcher_config,
        max_sequence_length=max_sequence_length,
        top_rows=top_rows,
        rare_token_threshold=rare_token_threshold,
        mlflow_db_path=mlflow_db_path,
    )
    reference = _collect_reference_diagnostics(
        dataset_name=reference_dataset_name,
        processed_directory=reference_processed_directory,
        encoded_directory=reference_encoded_directory,
        scale_matcher_config=scale_matcher_config,
        max_sequence_length=max_sequence_length,
        top_rows=top_rows,
        rare_token_threshold=rare_token_threshold,
    )
    summary = dict(primary.summary)
    tables = dict(primary.tables)
    if reference is not None:
        reference_rows = reference_comparison_rows(primary=primary, reference=reference)
        tables["reference_comparison"] = reference_rows
        summary["reference"] = _reference_summary(reference=reference, comparison_rows=reference_rows)

    output_directory.mkdir(parents=True, exist_ok=True)
    tables_directory = output_directory / TABLES_DIRECTORY_NAME
    table_paths = _write_tables(tables=tables, tables_directory=tables_directory)
    summary_path = output_directory / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=JSON_INDENT, sort_keys=True) + "\n", encoding="utf-8")
    report_path = output_directory / REPORT_NAME
    report_path.write_text(_render_markdown_report(summary=summary, table_paths=table_paths), encoding="utf-8")
    return DiagnosticReportResult(
        output_directory=output_directory,
        report_path=report_path,
        summary_path=summary_path,
        table_paths=table_paths,
    )


def collect_dataset_diagnostics(
    *,
    dataset_name: str,
    processed_directory: Path,
    encoded_directory: Path,
    scale_matcher_config: ScaleMatcherConfig,
    max_sequence_length: int,
    top_rows: int = DEFAULT_TOP_ROWS,
    rare_token_threshold: int = DEFAULT_RARE_TOKEN_THRESHOLD,
    mlflow_db_path: Path | None = None,
) -> CollectedDiagnostics:
    paths = _dataset_paths(
        dataset_name=dataset_name,
        processed_directory=processed_directory,
        encoded_directory=encoded_directory,
    )
    _require_diagnostic_inputs(paths)
    tokenizer_snapshot = load_tokenizer_snapshot_json(paths.tokenizer_snapshot_path)
    parsed_summary, parsed_tables = _parse_manifest_summary(paths.parsed_manifest_path)
    encoded_summary, encoded_tables = _encoded_manifest_summary(
        paths.encoded_manifest_path,
        max_sequence_length=max_sequence_length,
        top_rows=top_rows,
    )
    token_summary, token_tables = _token_jsonl_summary(
        paths.encoded_jsonl_path,
        tokenizer_snapshot=tokenizer_snapshot,
        rare_token_threshold=rare_token_threshold,
    )
    tonal_probe_rows = _tonal_probe_rows(scale_matcher_config)
    mlflow_summary, mlflow_tables = _mlflow_summary(
        mlflow_db_path=mlflow_db_path,
        dataset_name=dataset_name,
        tokenizer_hash=tokenizer_snapshot.tokenizer_hash,
    )
    summary: dict[str, JsonValue] = {
        "dataset": {
            "name": dataset_name,
            "processed_directory": paths.processed_directory.as_posix(),
            "encoded_directory": paths.encoded_directory.as_posix(),
            "parsed_manifest": paths.parsed_manifest_path.as_posix(),
            "encoded_manifest": paths.encoded_manifest_path.as_posix(),
            "encoded_jsonl": paths.encoded_jsonl_path.as_posix(),
        },
        "tokenizer": _tokenizer_summary(tokenizer_snapshot),
        "parsed": parsed_summary,
        "encoded": encoded_summary,
        "tokens": token_summary,
        "tonal_probes": cast(list[JsonValue], tonal_probe_rows),
        "mlflow": mlflow_summary,
    }
    tables = {
        **parsed_tables,
        **encoded_tables,
        **token_tables,
        "tonal_probes": tonal_probe_rows,
        **mlflow_tables,
    }
    return CollectedDiagnostics(paths=paths, summary=summary, tables=tables)


def reference_comparison_rows(
    *,
    primary: CollectedDiagnostics,
    reference: CollectedDiagnostics,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    primary_encoded = _dict_value(primary.summary["encoded"])
    reference_encoded = _dict_value(reference.summary["encoded"])
    primary_tokens = _dict_value(primary.summary["tokens"])
    reference_tokens = _dict_value(reference.summary["tokens"])
    rows.extend(
        _numeric_reference_rows(
            primary_encoded,
            reference_encoded,
            numeric_path=("numeric",),
            metric_names=tuple(field.value for field in _REFERENCE_NUMERIC_FIELDS),
        )
    )
    rows.extend(
        _simple_reference_rows(
            primary_encoded,
            reference_encoded,
            metric_names=(
                "parse_success_rate",
                "eligibility_rate",
                "encoded_sample_rate",
                "over_max_sequence_length_rate",
            ),
        )
    )
    rows.extend(
        _categorical_reference_rows(
            primary_encoded,
            reference_encoded,
            metric_names=("time_signature_distribution", "scale_type_distribution", "scale_root_distribution"),
        )
    )
    rows.extend(
        _categorical_reference_rows(
            primary_tokens,
            reference_tokens,
            metric_names=("token_kind_distribution", "degree_distribution", "accidental_distribution"),
        )
    )
    return rows


def _collect_reference_diagnostics(
    *,
    dataset_name: str | None,
    processed_directory: Path | None,
    encoded_directory: Path | None,
    scale_matcher_config: ScaleMatcherConfig,
    max_sequence_length: int,
    top_rows: int,
    rare_token_threshold: int,
) -> CollectedDiagnostics | None:
    if dataset_name is None or processed_directory is None or encoded_directory is None:
        return None

    return collect_dataset_diagnostics(
        dataset_name=dataset_name,
        processed_directory=processed_directory,
        encoded_directory=encoded_directory,
        scale_matcher_config=scale_matcher_config,
        max_sequence_length=max_sequence_length,
        top_rows=top_rows,
        rare_token_threshold=rare_token_threshold,
        mlflow_db_path=None,
    )


def _dataset_paths(
    *,
    dataset_name: str,
    processed_directory: Path,
    encoded_directory: Path,
) -> DatasetPaths:
    return DatasetPaths(
        dataset_name=dataset_name,
        processed_directory=processed_directory,
        encoded_directory=encoded_directory,
        parsed_manifest_path=processed_directory / PARSED_MANIFEST_NAME,
        encoded_manifest_path=encoded_directory / ENCODED_MANIFEST_NAME,
        encoded_jsonl_path=encoded_directory / ENCODED_JSONL_NAME,
        tokenizer_snapshot_path=encoded_directory / "tokenizer.json",
    )


def _require_diagnostic_inputs(paths: DatasetPaths) -> None:
    for path in (
        paths.parsed_manifest_path,
        paths.encoded_manifest_path,
        paths.encoded_jsonl_path,
        paths.tokenizer_snapshot_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"diagnostic input does not exist: {path}")


def _parse_manifest_summary(path: Path) -> tuple[dict[str, JsonValue], dict[str, list[CsvRow]]]:
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    time_signature_counts: Counter[str] = Counter()
    declared_key_counts: Counter[str] = Counter()
    parsed_files = 0
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            parsed_files += 1
            status = row.get(ParsedManifestField.STATUS.value, "")
            status_counts[status] += 1
            error_type = row.get(ParsedManifestField.ERROR_TYPE.value, "")
            if error_type:
                error_counts[error_type] += 1
            time_signature = row.get(ParsedManifestField.TIME_SIGNATURE.value, "")
            if time_signature:
                time_signature_counts[time_signature] += 1
            declared_key = row.get(ParsedManifestField.DECLARED_KEY_FIFTHS.value, "")
            if declared_key:
                declared_key_counts[declared_key] += 1

    successes = status_counts[ParsedManifestStatus.SUCCESS.value]
    errors = parsed_files - successes
    summary: dict[str, JsonValue] = {
        "files": parsed_files,
        "successes": successes,
        "errors": errors,
        "parse_success_rate": _rate(successes, parsed_files),
        "status_distribution": _counter_json(status_counts),
        "error_type_distribution": _counter_json(error_counts),
        "time_signature_distribution": _counter_json(time_signature_counts),
        "declared_key_fifths_distribution": _counter_json(declared_key_counts),
    }
    tables = {
        "parsed_error_types": _counter_rows(error_counts, denominator=parsed_files),
        "parsed_time_signatures": _counter_rows(time_signature_counts, denominator=successes),
        "parsed_declared_key_fifths": _counter_rows(declared_key_counts, denominator=successes),
    }
    return summary, tables


def _encoded_manifest_summary(
    path: Path,
    *,
    max_sequence_length: int,
    top_rows: int,
) -> tuple[dict[str, JsonValue], dict[str, list[CsvRow]]]:
    encoded = _EncodedManifestAccumulator(max_sequence_length=max_sequence_length, top_rows=top_rows)
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            encoded.add(row)

    return encoded.summary(), encoded.tables()


class _EncodedManifestAccumulator:
    def __init__(self, *, max_sequence_length: int, top_rows: int) -> None:
        self._max_sequence_length = max_sequence_length
        self._segments = 0
        self._eligible_segments = 0
        self._encoded_samples = 0
        self._over_max_sequence_length = 0
        self._missing_difficulty = 0
        self._categorical_counts: dict[str, Counter[str]] = {
            "time_signature_distribution": Counter(),
            "scale_type_distribution": Counter(),
            "scale_root_distribution": Counter(),
            "declared_key_fifths_distribution": Counter(),
            "spelling_key_fifths_distribution": Counter(),
            "spelling_context_source_distribution": Counter(),
        }
        self._scale_selection_counts: Counter[str] = Counter()
        self._ineligibility_counts: Counter[str] = Counter()
        self._boolean_counts: Counter[str] = Counter()
        self._numeric_values: dict[str, list[float]] = {field.value: [] for field in _NUMERIC_FIELDS}
        self._ambiguous_scale_rows = _TopRows(limit=top_rows)
        self._high_accidental_rows = _TopRows(limit=top_rows)
        self._low_in_scale_rows = _TopRows(limit=top_rows)
        self._low_margin_rows = _TopRows(limit=top_rows)
        self._long_token_rows = _TopRows(limit=top_rows)

    def add(self, row: dict[str, str]) -> None:
        self._segments += 1
        if _bool_value(row, EncodedManifestField.ELIGIBLE_FOR_TRAINING):
            self._eligible_segments += 1
        if _text_value(row, EncodedManifestField.ENCODED_LINE) != "":
            self._encoded_samples += 1
        if _text_value(row, EncodedManifestField.DIFFICULTY_LEVEL) == "":
            self._missing_difficulty += 1

        token_count = _float_value(row, EncodedManifestField.TOKEN_COUNT)
        if token_count is not None and token_count > self._max_sequence_length:
            self._over_max_sequence_length += 1

        for field in _NUMERIC_FIELDS:
            value = _float_value(row, field)
            if value is not None:
                self._numeric_values[field.value].append(value)

        for field in _BOOLEAN_FIELDS:
            if _bool_value(row, field):
                self._boolean_counts[field.value] += 1

        for field, count_name in (
            (EncodedManifestField.TIME_SIGNATURE, "time_signature_distribution"),
            (EncodedManifestField.SCALE_TYPE, "scale_type_distribution"),
            (EncodedManifestField.SCALE_ROOT, "scale_root_distribution"),
            (EncodedManifestField.DECLARED_KEY_FIFTHS, "declared_key_fifths_distribution"),
            (EncodedManifestField.SPELLING_KEY_FIFTHS, "spelling_key_fifths_distribution"),
            (EncodedManifestField.SPELLING_CONTEXT_SOURCE, "spelling_context_source_distribution"),
        ):
            text = _text_value(row, field)
            if text:
                self._categorical_counts[count_name][_manifest_label(field, text)] += 1

        scale_label = _scale_selection_label(row)
        if scale_label:
            self._scale_selection_counts[scale_label] += 1

        for reason in _ineligibility_reasons(row):
            self._ineligibility_counts[reason] += 1

        self._add_outlier_rows(row)

    def summary(self) -> dict[str, JsonValue]:
        numeric_summary: dict[str, JsonValue] = {
            field_name: cast(JsonValue, _numeric_summary(values))
            for field_name, values in self._numeric_values.items()
            if values
        }
        summary: dict[str, JsonValue] = {
            "segments": self._segments,
            "eligible_segments": self._eligible_segments,
            "encoded_samples": self._encoded_samples,
            "eligibility_rate": _rate(self._eligible_segments, self._segments),
            "encoded_sample_rate": _rate(self._encoded_samples, self._segments),
            "missing_difficulty_labels": self._missing_difficulty,
            "missing_difficulty_label_rate": _rate(self._missing_difficulty, self._segments),
            "max_sequence_length": self._max_sequence_length,
            "over_max_sequence_length": self._over_max_sequence_length,
            "over_max_sequence_length_rate": _rate(self._over_max_sequence_length, self._segments),
            "ineligibility_reasons": _counter_json(self._ineligibility_counts),
            "boolean_rates": {
                field.value: _rate(self._boolean_counts[field.value], self._segments) for field in _BOOLEAN_FIELDS
            },
            "numeric": numeric_summary,
        }
        for count_name, counter in self._categorical_counts.items():
            summary[count_name] = _counter_json(counter)
        summary["scale_selection_distribution"] = _counter_json(self._scale_selection_counts)
        return summary

    def tables(self) -> dict[str, list[CsvRow]]:
        numeric_rows = [
            _numeric_summary_row(field_name, values) for field_name, values in self._numeric_values.items() if values
        ]
        return {
            "encoded_numeric_summary": numeric_rows,
            "ineligibility_reasons": _counter_rows(self._ineligibility_counts, denominator=self._segments),
            "scale_types": _counter_rows(
                self._categorical_counts["scale_type_distribution"],
                denominator=self._segments,
            ),
            "scale_roots": _counter_rows(
                self._categorical_counts["scale_root_distribution"],
                denominator=self._segments,
            ),
            "time_signatures": _counter_rows(
                self._categorical_counts["time_signature_distribution"],
                denominator=self._segments,
            ),
            "declared_key_fifths": _counter_rows(
                self._categorical_counts["declared_key_fifths_distribution"],
                denominator=self._segments,
            ),
            "spelling_key_fifths": _counter_rows(
                self._categorical_counts["spelling_key_fifths_distribution"],
                denominator=self._segments,
            ),
            "spelling_context_sources": _counter_rows(
                self._categorical_counts["spelling_context_source_distribution"],
                denominator=self._segments,
            ),
            "declared_key_vs_selected_scale": _counter_rows(
                self._scale_selection_counts,
                denominator=self._segments,
            ),
            "outliers_ambiguous_scale": self._ambiguous_scale_rows.rows(),
            "outliers_high_accidental": self._high_accidental_rows.rows(),
            "outliers_low_in_scale": self._low_in_scale_rows.rows(),
            "outliers_low_margin": self._low_margin_rows.rows(),
            "outliers_long_tokens": self._long_token_rows.rows(),
        }

    def _add_outlier_rows(self, row: dict[str, str]) -> None:
        output_row = _outlier_row(row)
        token_count = _float_value(row, EncodedManifestField.TOKEN_COUNT)
        accidental_fraction = _float_value(row, EncodedManifestField.ACCIDENTAL_NOTE_FRACTION)
        in_scale_fraction = _float_value(row, EncodedManifestField.IN_SCALE_NOTE_FRACTION)
        best_margin = _float_value(row, EncodedManifestField.SCALE_MATCH_BEST_MARGIN)
        tied_best = _float_value(row, EncodedManifestField.SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT)
        ambiguous = _bool_value(row, EncodedManifestField.SCALE_MATCH_AMBIGUOUS)
        self._long_token_rows.add(score=token_count, row=output_row)
        self._high_accidental_rows.add(score=accidental_fraction, row=output_row)
        self._low_in_scale_rows.add(
            score=None if in_scale_fraction is None else 1.0 - in_scale_fraction,
            row=output_row,
        )
        self._low_margin_rows.add(score=None if best_margin is None else -best_margin, row=output_row)
        if ambiguous or (tied_best is not None and tied_best > 1):
            self._ambiguous_scale_rows.add(score=(tied_best or 0.0) + (1.0 if ambiguous else 0.0), row=output_row)


def _token_jsonl_summary(
    path: Path,
    *,
    tokenizer_snapshot: TokenizerSnapshot,
    rare_token_threshold: int,
) -> tuple[dict[str, JsonValue], dict[str, list[CsvRow]]]:
    tokenization_config = TokenizationConfig.model_validate(tokenizer_snapshot.tokenization_config)
    vocabulary = TokenVocabulary(DurationVocabulary(tokenization_config))
    token_id_counts: Counter[int] = Counter()
    sample_count = 0
    bar_position_mismatch_count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = cast(dict[str, JsonValue], json.loads(line))
            token_ids = _json_int_list(payload, "token_ids")
            bar_positions = _json_int_list(payload, "bar_positions")
            sample_count += 1
            if len(token_ids) != len(bar_positions):
                bar_position_mismatch_count += 1
            token_id_counts.update(token_ids)

    token_kind_counts: Counter[str] = Counter()
    degree_counts: Counter[str] = Counter()
    accidental_counts: Counter[str] = Counter()
    octave_counts: Counter[str] = Counter()
    note_duration_counts: Counter[str] = Counter()
    rest_duration_counts: Counter[str] = Counter()
    hold_duration_counts: Counter[str] = Counter()
    for token_id, count in token_id_counts.items():
        token = vocabulary.id_to_token(token_id)
        match token:
            case NoteToken():
                token_kind_counts["note"] += count
                degree_counts[str(token.degree)] += count
                accidental_counts[str(token.accidental)] += count
                octave_counts[str(token.octave_offset)] += count
                note_duration_counts[str(token.duration_id)] += count
            case RestToken():
                token_kind_counts["rest"] += count
                rest_duration_counts[str(token.duration_id)] += count
            case HoldToken():
                token_kind_counts["hold"] += count
                hold_duration_counts[str(token.duration_id)] += count
            case BarToken():
                token_kind_counts["bar"] += count
            case EndToken():
                token_kind_counts["end"] += count
            case HandToken(hand=Hand.RIGHT):
                token_kind_counts["right_hand"] += count
            case HandToken(hand=Hand.LEFT):
                token_kind_counts["left_hand"] += count
            case JoinWithPreviousToken():
                token_kind_counts["join_with_previous"] += count
            case StartToken():
                token_kind_counts["start"] += count

    token_count = sum(token_id_counts.values())
    rare_token_count = sum(1 for count in token_id_counts.values() if count <= rare_token_threshold)
    summary: dict[str, JsonValue] = {
        "samples": sample_count,
        "tokens": token_count,
        "bar_position_mismatches": bar_position_mismatch_count,
        "active_token_ids": len(token_id_counts),
        "vocabulary_size": tokenizer_snapshot.vocabulary_size,
        "active_token_id_fraction": _rate(len(token_id_counts), tokenizer_snapshot.vocabulary_size),
        "rare_token_threshold": rare_token_threshold,
        "rare_token_ids": rare_token_count,
        "token_entropy_bits": _entropy_bits(token_id_counts),
        "token_kind_distribution": _counter_json(token_kind_counts),
        "degree_distribution": _counter_json(degree_counts),
        "accidental_distribution": _counter_json(accidental_counts),
        "octave_offset_distribution": _counter_json(octave_counts),
        "note_duration_distribution": _counter_json(note_duration_counts),
        "rest_duration_distribution": _counter_json(rest_duration_counts),
        "hold_duration_distribution": _counter_json(hold_duration_counts),
    }
    tables = {
        "token_kinds": _counter_rows(token_kind_counts, denominator=token_count),
        "token_ids_top": _token_id_rows(
            token_id_counts,
            denominator=token_count,
            limit=DEFAULT_TOKEN_ID_TABLE_LIMIT,
            vocabulary=vocabulary,
        ),
        "token_attributes": _token_attribute_rows(
            counters={
                "degree": degree_counts,
                "accidental": accidental_counts,
                "octave_offset": octave_counts,
                "note_duration_id": note_duration_counts,
                "rest_duration_id": rest_duration_counts,
                "hold_duration_id": hold_duration_counts,
            }
        ),
    }
    return summary, tables


def _tonal_probe_rows(scale_matcher_config: ScaleMatcherConfig) -> list[CsvRow]:
    return [
        _tonal_probe_row(
            label="A natural minor",
            pitch_classes=_A_NATURAL_MINOR_PITCH_CLASSES,
            declared_key_fifths=_DECLARED_C_MAJOR_FIFTHS,
            scale_matcher_config=scale_matcher_config,
        ),
        _tonal_probe_row(
            label="A harmonic minor",
            pitch_classes=_A_HARMONIC_MINOR_PITCH_CLASSES,
            declared_key_fifths=_DECLARED_C_MAJOR_FIFTHS,
            scale_matcher_config=scale_matcher_config,
        ),
        _tonal_probe_row(
            label="A melodic minor",
            pitch_classes=_A_MELODIC_MINOR_PITCH_CLASSES,
            declared_key_fifths=_DECLARED_C_MAJOR_FIFTHS,
            scale_matcher_config=scale_matcher_config,
        ),
    ]


def _tonal_probe_row(
    *,
    label: str,
    pitch_classes: tuple[int, ...],
    declared_key_fifths: int,
    scale_matcher_config: ScaleMatcherConfig,
) -> CsvRow:
    histogram = {pitch_class: Fraction(1, len(pitch_classes)) for pitch_class in pitch_classes}
    scale_match = match_scale_histogram(
        histogram,
        declared_key_fifths=declared_key_fifths,
        config=scale_matcher_config,
    )
    tonic_degree = pitch_to_degree(
        _A4_MIDI_PITCH,
        scale_root=scale_match.scale_root,
        key_fifths=declared_key_fifths,
        scale_type=scale_match.scale_type,
        hand=Hand.RIGHT,
    )
    return {
        "probe": label,
        "selected_scale_root": scale_match.scale_root,
        "selected_scale_root_name": KEYS[scale_match.scale_root],
        "selected_scale_type": scale_match.scale_type.value,
        "reference_pitch": "A4",
        "reference_pitch_degree": tonic_degree.degree,
        "reference_pitch_accidental": tonic_degree.accidental,
        "reference_pitch_maps_to_degree_1": tonic_degree.degree == 1 and tonic_degree.accidental == 0,
        "best_margin": scale_match.diagnostics.best_margin,
        "tied_best_candidate_count": scale_match.diagnostics.tied_best_candidate_count,
        "declared_match_used": scale_match.diagnostics.declared_match_used,
    }


def _tokenizer_summary(tokenizer_snapshot: TokenizerSnapshot) -> dict[str, JsonValue]:
    return {
        "schema_version": list(tokenizer_snapshot.schema_version),
        "tokenizer_hash": tokenizer_snapshot.tokenizer_hash,
        "vocabulary_size": tokenizer_snapshot.vocabulary_size,
        "duration_count": len(tokenizer_snapshot.duration_fractions),
        "special_token_ids": {key.value: value for key, value in tokenizer_snapshot.special_token_ids.items()},
        "tokenization_config": cast(dict[str, JsonValue], tokenizer_snapshot.tokenization_config),
    }


def _mlflow_summary(
    *,
    mlflow_db_path: Path | None,
    dataset_name: str,
    tokenizer_hash: str,
) -> tuple[dict[str, JsonValue], dict[str, list[CsvRow]]]:
    if mlflow_db_path is None:
        return {"enabled": False, "reason": "no MLflow database configured"}, {}
    if not mlflow_db_path.is_file():
        return {"enabled": False, "reason": f"MLflow database not found: {mlflow_db_path.as_posix()}"}, {}

    try:
        with sqlite3.connect(mlflow_db_path) as connection:
            connection.row_factory = sqlite3.Row
            process_runs = _mlflow_runs(
                connection,
                experiment_name="musak-process",
                dataset_name=dataset_name,
                tokenizer_hash=tokenizer_hash,
                limit=5,
            )
            training_runs = _mlflow_runs(
                connection,
                experiment_name="musak-pretrain",
                dataset_name=None,
                tokenizer_hash=None,
                limit=5,
            )
    except sqlite3.Error as exception:
        return {"enabled": False, "reason": f"MLflow lookup failed: {exception}"}, {}

    summary: dict[str, JsonValue] = {
        "enabled": True,
        "database": mlflow_db_path.as_posix(),
        "process_runs": cast(list[JsonValue], process_runs),
        "training_runs": cast(list[JsonValue], training_runs),
    }
    return summary, {"mlflow_process_runs": process_runs, "mlflow_training_runs": training_runs}


def _mlflow_runs(
    connection: sqlite3.Connection,
    *,
    experiment_name: str,
    dataset_name: str | None,
    tokenizer_hash: str | None,
    limit: int,
) -> list[CsvRow]:
    conditions = ["e.name = ?"]
    parameters: list[str | int] = [experiment_name]
    if dataset_name is not None:
        conditions.append(
            "exists (select 1 from params p where p.run_uuid = r.run_uuid "
            "and p.key = 'data.dataset_name' and p.value = ?)"
        )
        parameters.append(dataset_name)
    if tokenizer_hash is not None:
        conditions.append(
            "exists (select 1 from params p where p.run_uuid = r.run_uuid "
            "and p.key = 'data.tokenizer_hash' and p.value = ?)"
        )
        parameters.append(tokenizer_hash)
    parameters.append(limit)
    query = f"""
        select
            r.run_uuid,
            r.name,
            r.status,
            datetime(r.start_time / 1000, 'unixepoch') as start_utc,
            datetime(r.end_time / 1000, 'unixepoch') as end_utc
        from runs r
        join experiments e on e.experiment_id = r.experiment_id
        where {" and ".join(conditions)}
        order by r.start_time desc
        limit ?
    """
    rows = []
    for row in connection.execute(query, parameters):
        run_uuid = str(row["run_uuid"])
        rows.append(
            {
                "run_uuid": run_uuid,
                "name": str(row["name"]),
                "status": str(row["status"]),
                "start_utc": str(row["start_utc"]),
                "end_utc": "" if row["end_utc"] is None else str(row["end_utc"]),
                **_mlflow_key_metrics(connection, run_uuid=run_uuid),
            }
        )
    return rows


def _mlflow_key_metrics(connection: sqlite3.Connection, *, run_uuid: str) -> CsvRow:
    metric_keys = (
        "dataset/overall/count/encoded_samples",
        "dataset/overall/rate/eligible",
        "dataset/scale_match/mean/best_margin",
        "dataset/diagnostics/mean/accidental_note_fraction",
        "model/validation/mean/loss",
        "model/validation/rate/token_accuracy",
        "model/validation/rate/token_kind_accuracy",
        "generation/hard/rate/constraint_failure",
        "generation/soft/rate/constraint_failure",
        "generation/figure/mean/identity_total_variation_distance",
    )
    placeholders = ",".join("?" for _ in metric_keys)
    query = f"""
        select key, value
        from latest_metrics
        where run_uuid = ? and key in ({placeholders})
    """
    output: CsvRow = {}
    for row in connection.execute(query, [run_uuid, *metric_keys]):
        output[str(row["key"])] = float(row["value"])
    return output


def _reference_summary(
    *,
    reference: CollectedDiagnostics,
    comparison_rows: list[CsvRow],
) -> dict[str, JsonValue]:
    return {
        "name": reference.paths.dataset_name,
        "processed_directory": reference.paths.processed_directory.as_posix(),
        "encoded_directory": reference.paths.encoded_directory.as_posix(),
        "comparison_rows": cast(list[JsonValue], comparison_rows),
    }


def _numeric_reference_rows(
    primary: dict[str, JsonValue],
    reference: dict[str, JsonValue],
    *,
    numeric_path: tuple[str, ...],
    metric_names: tuple[str, ...],
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    primary_numeric = _nested_dict(primary, numeric_path)
    reference_numeric = _nested_dict(reference, numeric_path)
    for metric_name in metric_names:
        primary_value = _nested_mean(primary_numeric, metric_name)
        reference_value = _nested_mean(reference_numeric, metric_name)
        if primary_value is None or reference_value is None:
            continue
        rows.append(_comparison_row(metric_name, primary_value, reference_value, comparison_type="mean_delta"))
    return rows


def _simple_reference_rows(
    primary: dict[str, JsonValue],
    reference: dict[str, JsonValue],
    *,
    metric_names: tuple[str, ...],
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for metric_name in metric_names:
        primary_value = _json_float(primary.get(metric_name))
        reference_value = _json_float(reference.get(metric_name))
        if primary_value is None or reference_value is None:
            continue
        rows.append(_comparison_row(metric_name, primary_value, reference_value, comparison_type="delta"))
    return rows


def _categorical_reference_rows(
    primary: dict[str, JsonValue],
    reference: dict[str, JsonValue],
    *,
    metric_names: tuple[str, ...],
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for metric_name in metric_names:
        primary_counter = _json_counter(primary.get(metric_name))
        reference_counter = _json_counter(reference.get(metric_name))
        if not primary_counter and not reference_counter:
            continue
        rows.append(
            {
                "metric": metric_name,
                "comparison_type": "total_variation_distance",
                "primary": _total_variation_distance(primary_counter, reference_counter),
                "reference": 0.0,
                "delta": _total_variation_distance(primary_counter, reference_counter),
                "ratio": "",
            }
        )
    return rows


def _comparison_row(metric_name: str, primary_value: float, reference_value: float, *, comparison_type: str) -> CsvRow:
    return {
        "metric": metric_name,
        "comparison_type": comparison_type,
        "primary": primary_value,
        "reference": reference_value,
        "delta": primary_value - reference_value,
        "ratio": "" if reference_value == 0 else primary_value / reference_value,
    }


def _write_tables(*, tables: dict[str, list[CsvRow]], tables_directory: Path) -> dict[str, Path]:
    table_paths: dict[str, Path] = {}
    for name, rows in tables.items():
        path = tables_directory / f"{name}.csv"
        columns = _columns_for_rows(rows)
        write_csv_rows(path, columns=columns, rows=rows)
        table_paths[name] = path
    return table_paths


def _render_markdown_report(*, summary: dict[str, JsonValue], table_paths: dict[str, Path]) -> str:
    dataset = _dict_value(summary["dataset"])
    parsed = _dict_value(summary["parsed"])
    encoded = _dict_value(summary["encoded"])
    tokens = _dict_value(summary["tokens"])
    tokenizer = _dict_value(summary["tokenizer"])
    lines = [
        f"# Dataset Diagnostic Report: {dataset['name']}",
        "",
        "## Inputs",
        "",
        f"- Processed directory: `{dataset['processed_directory']}`",
        f"- Encoded directory: `{dataset['encoded_directory']}`",
        f"- Tokenizer hash: `{tokenizer['tokenizer_hash']}`",
        f"- Vocabulary size: `{tokenizer['vocabulary_size']}`",
        "",
        "## Overview",
        "",
        _markdown_table(
            [
                ("Parsed files", parsed["files"]),
                ("Parse success rate", _format_percent_value(parsed["parse_success_rate"])),
                ("Segments", encoded["segments"]),
                ("Eligible segments", encoded["eligible_segments"]),
                ("Eligibility rate", _format_percent_value(encoded["eligibility_rate"])),
                ("Encoded samples", encoded["encoded_samples"]),
                ("Over max sequence length", encoded["over_max_sequence_length"]),
                ("Active token IDs", tokens["active_token_ids"]),
                ("Token entropy bits", _format_float(tokens["token_entropy_bits"])),
            ]
        ),
        "",
        "## Tonal Probes",
        "",
        _markdown_rows(cast(list[dict[str, JsonValue]], summary["tonal_probes"]), limit=8),
        "",
        "## Key Tables",
        "",
    ]
    for table_name in sorted(table_paths):
        lines.append(f"- `{table_name}`: `{table_paths[table_name].as_posix()}`")
    if "reference" in summary:
        lines.extend(["", "## Reference Comparison", "", "See `tables/reference_comparison.csv`."])
    mlflow = _dict_value(summary["mlflow"])
    if bool(mlflow.get("enabled")):
        lines.extend(
            ["", "## MLflow", "", "See `tables/mlflow_process_runs.csv` and `tables/mlflow_training_runs.csv`."]
        )
    return "\n".join(lines) + "\n"


def _markdown_table(rows: list[tuple[str, JsonValue]]) -> str:
    output = ["| Metric | Value |", "| --- | --- |"]
    output.extend(f"| {metric} | {value} |" for metric, value in rows)
    return "\n".join(output)


def _markdown_rows(rows: list[dict[str, JsonValue]], *, limit: int) -> str:
    if not rows:
        return "(none)"

    columns = list(rows[0].keys())
    output = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        output.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(output)


def _text_value(row: dict[str, str], field: EncodedManifestField) -> str:
    return row.get(field.value, "").strip()


def _float_value(row: dict[str, str], field: EncodedManifestField) -> float | None:
    value = _text_value(row, field)
    if value == "":
        return None
    return float(value)


def _bool_value(row: dict[str, str], field: EncodedManifestField) -> bool:
    return _text_value(row, field) == "True"


def _manifest_label(field: EncodedManifestField, value: str) -> str:
    if field == EncodedManifestField.SCALE_ROOT:
        return f"{value}:{KEYS.get(int(value), value)}"
    return value


def _scale_selection_label(row: dict[str, str]) -> str:
    declared = _text_value(row, EncodedManifestField.DECLARED_KEY_FIFTHS) or "(none)"
    scale_root = _text_value(row, EncodedManifestField.SCALE_ROOT)
    scale_type = _text_value(row, EncodedManifestField.SCALE_TYPE)
    if not scale_root or not scale_type:
        return ""
    scale_root_label = _manifest_label(EncodedManifestField.SCALE_ROOT, scale_root)
    return f"declared={declared} selected={scale_root_label}/{scale_type}"


def _ineligibility_reasons(row: dict[str, str]) -> list[str]:
    reasons = _text_value(row, EncodedManifestField.INELIGIBILITY_REASONS)
    if reasons == "":
        return []
    return [reason for reason in reasons.split(_REASON_SEPARATOR) if reason]


def _outlier_row(row: dict[str, str]) -> CsvRow:
    return {field.value: row.get(field.value, "") for field in _OUTLIER_FIELDS}


def _counter_json(counter: Counter[str]) -> dict[str, JsonValue]:
    return {key: count for key, count in counter.most_common()}


def _counter_rows(counter: Counter[str], *, denominator: int) -> list[CsvRow]:
    return [
        {
            _VALUE_COLUMN: key,
            _COUNT_COLUMN: count,
            _PERCENT_COLUMN: _rate(count, denominator),
        }
        for key, count in counter.most_common()
    ]


def _numeric_summary(values: list[float]) -> dict[str, JsonScalar]:
    sorted_values = sorted(values)
    total = sum(sorted_values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p10": _quantile(sorted_values, 0.10),
        "p25": _quantile(sorted_values, 0.25),
        "p50": _quantile(sorted_values, 0.50),
        "mean": total / len(sorted_values),
        "p75": _quantile(sorted_values, 0.75),
        "p95": _quantile(sorted_values, 0.95),
        "max": sorted_values[-1],
    }


def _numeric_summary_row(field_name: str, values: list[float]) -> CsvRow:
    summary = _numeric_summary(values)
    row: CsvRow = {"metric": field_name}
    row.update(summary)
    return row


def _quantile(sorted_values: list[float], fraction: float) -> float:
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _entropy_bits(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def _token_id_rows(
    counter: Counter[int],
    *,
    denominator: int,
    limit: int,
    vocabulary: TokenVocabulary,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for token_id, count in counter.most_common(limit):
        rows.append(
            {
                "token_id": token_id,
                "token": repr(vocabulary.id_to_token(token_id)),
                _COUNT_COLUMN: count,
                _PERCENT_COLUMN: _rate(count, denominator),
            }
        )
    return rows


def _token_attribute_rows(*, counters: dict[str, Counter[str]]) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for attribute, counter in counters.items():
        denominator = sum(counter.values())
        for value, count in counter.most_common():
            rows.append(
                {
                    "attribute": attribute,
                    _VALUE_COLUMN: value,
                    _COUNT_COLUMN: count,
                    _PERCENT_COLUMN: _rate(count, denominator),
                }
            )
    return rows


def _json_int_list(payload: dict[str, JsonValue], key: str) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"expected JSON list field {key}")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"expected integer values in JSON list field {key}")
    return cast(list[int], value)


def _columns_for_rows(rows: list[CsvRow]) -> tuple[str, ...]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    return tuple(columns or ("value",))


def _dict_value(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError("expected JSON object")
    return value


def _nested_dict(value: dict[str, JsonValue], path: tuple[str, ...]) -> dict[str, JsonValue]:
    current = value
    for part in path:
        current = _dict_value(current[part])
    return current


def _nested_mean(value: dict[str, JsonValue], metric_name: str) -> float | None:
    metric = value.get(metric_name)
    if not isinstance(metric, dict):
        return None
    return _json_float(metric.get("mean"))


def _json_float(value: JsonValue | None) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _json_counter(value: JsonValue | None) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counter
    for key, item in value.items():
        if isinstance(item, int | float):
            counter[key] = int(item)
    return counter


def _total_variation_distance(primary: Counter[str], reference: Counter[str]) -> float:
    primary_total = sum(primary.values())
    reference_total = sum(reference.values())
    if primary_total == 0 and reference_total == 0:
        return 0.0
    values = set(primary) | set(reference)
    distance = 0.0
    for value in values:
        primary_probability = primary[value] / primary_total if primary_total else 0.0
        reference_probability = reference[value] / reference_total if reference_total else 0.0
        distance += abs(primary_probability - reference_probability)
    return 0.5 * distance


def _format_percent_value(value: JsonValue) -> str:
    number = _json_float(value)
    if number is None:
        return ""
    return f"{number * 100:.2f}%"


def _format_float(value: JsonValue) -> str:
    number = _json_float(value)
    if number is None:
        return ""
    return f"{number:.4f}"
