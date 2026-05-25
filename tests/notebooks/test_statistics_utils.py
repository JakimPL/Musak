from pathlib import Path

import pytest

from musak_model.processing.manifest import EncodedManifestField, ParsedManifestField
from notebooks.utils.statistics import (
    COUNT_COLUMN,
    VALUE_COLUMN,
    DatasetStatistics,
    diagnostic_bucket_distribution,
    diagnostic_summary_rows,
    eligibility_distribution,
    encoded_table_frame,
    ineligibility_reason_distribution,
    load_dataset_statistics,
    overview_rows,
    parse_error_table_frame,
    parsed_table_frame,
    read_encoded_manifest_frame,
    reason_by_column,
    scale_root_distribution,
    selected_table_row,
    table_records,
    token_histogram_distribution,
    token_summary_rows,
)


def test_load_dataset_statistics_requires_current_encoded_manifest_columns(tmp_path: Path) -> None:
    encoded_path = tmp_path / "encoded.csv"
    encoded_path.write_text("segment_id,source_id\nabc,source\n", encoding="utf-8")

    with pytest.raises(ValueError, match="token_count"):
        read_encoded_manifest_frame(encoded_path)


def test_dataset_overview_counts_parsed_and_encoded_rows(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "PDMX"
    encoded_directory = dataset_dir / "encoded" / "abc"
    encoded_directory.mkdir(parents=True)
    _write_csv(
        dataset_dir / "parsed.csv",
        ParsedManifestField,
        [
            {ParsedManifestField.SOURCE_ID: "a", ParsedManifestField.STATUS: "success"},
            {ParsedManifestField.SOURCE_ID: "b", ParsedManifestField.STATUS: "error"},
        ],
    )
    _write_csv(
        encoded_directory / "encoded.csv",
        EncodedManifestField,
        [
            {
                EncodedManifestField.SEGMENT_ID: "a-0",
                EncodedManifestField.ELIGIBLE_FOR_TRAINING: "True",
                EncodedManifestField.ENCODED_LINE: "0",
                EncodedManifestField.TOKEN_COUNT: "12",
            },
            {
                EncodedManifestField.SEGMENT_ID: "b-0",
                EncodedManifestField.ELIGIBLE_FOR_TRAINING: "False",
                EncodedManifestField.TOKEN_COUNT: "8",
            },
        ],
    )

    stats = load_dataset_statistics(dataset_dir, encoded_directory)

    assert overview_rows(stats) == [
        {"Metric": "Parsed files", "Value": "2"},
        {"Metric": "Parsed successfully", "Value": "1"},
        {"Metric": "Parse errors", "Value": "1"},
        {"Metric": "Parse success rate", "Value": "50.0%"},
        {"Metric": "Segments", "Value": "2"},
        {"Metric": "Eligible segments", "Value": "1"},
        {"Metric": "Eligibility rate", "Value": "50.0%"},
        {"Metric": "Encoded samples", "Value": "1"},
    ]


def test_encoded_statistics_expand_reasons_and_token_summary(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "PDMX"
    encoded_directory = dataset_dir / "encoded" / "abc"
    encoded_directory.mkdir(parents=True)
    _write_csv(dataset_dir / "parsed.csv", ParsedManifestField, [{ParsedManifestField.SOURCE_ID: "a"}])
    _write_csv(
        encoded_directory / "encoded.csv",
        EncodedManifestField,
        [
            {
                EncodedManifestField.SEGMENT_ID: "a-0",
                EncodedManifestField.ELIGIBLE_FOR_TRAINING: "False",
                EncodedManifestField.INELIGIBILITY_REASONS: "quantization_error|overlapping_events",
                EncodedManifestField.TIME_SIGNATURE: "4/4",
                EncodedManifestField.TOKEN_COUNT: "10",
            },
            {
                EncodedManifestField.SEGMENT_ID: "a-1",
                EncodedManifestField.ELIGIBLE_FOR_TRAINING: "False",
                EncodedManifestField.INELIGIBILITY_REASONS: "quantization_error",
                EncodedManifestField.TIME_SIGNATURE: "3/4",
                EncodedManifestField.TOKEN_COUNT: "30",
            },
            {
                EncodedManifestField.SEGMENT_ID: "a-2",
                EncodedManifestField.ELIGIBLE_FOR_TRAINING: "True",
                EncodedManifestField.TOKEN_COUNT: "20",
            },
        ],
    )
    stats = load_dataset_statistics(dataset_dir, encoded_directory)
    assert stats.encoded is not None

    eligibility = eligibility_distribution(stats.encoded)
    reasons = ineligibility_reason_distribution(stats.encoded)
    reason_time = reason_by_column(stats.encoded, EncodedManifestField.TIME_SIGNATURE)
    token_rows = token_summary_rows(stats.encoded)
    token_distribution = token_histogram_distribution(stats.encoded, bins=2)

    assert eligibility.set_index(VALUE_COLUMN).loc["ineligible", COUNT_COLUMN] == 2
    assert reasons.set_index(VALUE_COLUMN).loc["quantization_error", COUNT_COLUMN] == 2
    assert len(reason_time) == 3
    assert all(isinstance(key, str) for row in table_records(reason_time) for key in row)
    assert int(token_distribution[COUNT_COLUMN].sum()) == 3
    assert len(token_distribution) <= 4
    assert token_rows[0] == {"Metric": "min", "Value": "10"}
    assert token_rows[2] == {"Metric": "median", "Value": "20"}
    assert token_rows[-1] == {"Metric": "max", "Value": "30"}


def test_diagnostic_statistics_parse_and_summarize_encoded_manifest(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "PDMX"
    encoded_directory = dataset_dir / "encoded" / "abc"
    encoded_directory.mkdir(parents=True)
    _write_csv(dataset_dir / "parsed.csv", ParsedManifestField, [{ParsedManifestField.SOURCE_ID: "a"}])
    _write_csv(
        encoded_directory / "encoded.csv",
        EncodedManifestField,
        [
            {
                EncodedManifestField.SEGMENT_ID: "a-0",
                EncodedManifestField.EMPTY_SCORE: "True",
                EncodedManifestField.ONE_HAND_ONLY: "False",
                EncodedManifestField.RIGHT_SILENCE_FRACTION: "1.0",
                EncodedManifestField.LEFT_SILENCE_FRACTION: "1.0",
                EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION: "1.0",
                EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION: "0.0",
                EncodedManifestField.HAND_ACTIVITY_BALANCE: "1.0",
                EncodedManifestField.NOTE_TOKEN_FRACTION: "0.0",
                EncodedManifestField.REST_TOKEN_FRACTION: "0.5",
            },
            {
                EncodedManifestField.SEGMENT_ID: "a-1",
                EncodedManifestField.EMPTY_SCORE: "False",
                EncodedManifestField.ONE_HAND_ONLY: "True",
                EncodedManifestField.RIGHT_SILENCE_FRACTION: "0.5",
                EncodedManifestField.LEFT_SILENCE_FRACTION: "1.0",
                EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION: "0.5",
                EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION: "0.0",
                EncodedManifestField.HAND_ACTIVITY_BALANCE: "0.0",
                EncodedManifestField.NOTE_TOKEN_FRACTION: "0.5",
                EncodedManifestField.REST_TOKEN_FRACTION: "0.0",
            },
        ],
    )
    stats = load_dataset_statistics(dataset_dir, encoded_directory)
    assert stats.encoded is not None

    summary = diagnostic_summary_rows(stats.encoded)
    buckets = diagnostic_bucket_distribution(stats.encoded, EncodedManifestField.RIGHT_SILENCE_FRACTION, bins=2)

    assert stats.encoded[EncodedManifestField.EMPTY_SCORE].tolist() == [True, False]
    assert {"Metric": "empty score rate", "Value": "50.0%"} in summary
    assert {"Metric": "right silence", "Value": "75.0%"} in summary
    assert int(buckets[COUNT_COLUMN].sum()) == 2


def test_scale_root_distribution_maps_pitch_classes_to_names(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "PDMX"
    encoded_directory = dataset_dir / "encoded" / "abc"
    encoded_directory.mkdir(parents=True)
    _write_csv(dataset_dir / "parsed.csv", ParsedManifestField, [{ParsedManifestField.SOURCE_ID: "a"}])
    _write_csv(
        encoded_directory / "encoded.csv",
        EncodedManifestField,
        [
            {EncodedManifestField.SEGMENT_ID: "a-0", EncodedManifestField.SCALE_ROOT: "0"},
            {EncodedManifestField.SEGMENT_ID: "a-1", EncodedManifestField.SCALE_ROOT: "1"},
            {EncodedManifestField.SEGMENT_ID: "a-2", EncodedManifestField.SCALE_ROOT: "10"},
        ],
    )
    stats = load_dataset_statistics(dataset_dir, encoded_directory)
    assert stats.encoded is not None

    distribution = scale_root_distribution(stats.encoded, EncodedManifestField.SCALE_ROOT, top_n=12)

    assert set(distribution[VALUE_COLUMN]) == {"C", "C#", "A#"}


def test_overview_supports_parsed_only_statistics(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "PDMX"
    dataset_dir.mkdir()
    _write_csv(
        dataset_dir / "parsed.csv",
        ParsedManifestField,
        [{ParsedManifestField.SOURCE_ID: "a", ParsedManifestField.STATUS: "success"}],
    )

    stats = DatasetStatistics(parsed=load_dataset_statistics(dataset_dir, None).parsed, encoded=None)

    assert stats.has_encoded is False
    assert overview_rows(stats)[0] == {"Metric": "Parsed files", "Value": "1"}


def test_manifest_table_frames_are_not_truncated(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "PDMX"
    dataset_dir.mkdir()
    _write_csv(
        dataset_dir / "parsed.csv",
        ParsedManifestField,
        [
            {ParsedManifestField.SOURCE_ID: "a", ParsedManifestField.STATUS: "error"},
            {ParsedManifestField.SOURCE_ID: "b", ParsedManifestField.STATUS: "success"},
            {ParsedManifestField.SOURCE_ID: "c", ParsedManifestField.STATUS: "error"},
        ],
    )

    stats = load_dataset_statistics(dataset_dir, None)

    assert len(parsed_table_frame(stats.parsed)) == 3
    assert len(parse_error_table_frame(stats.parsed)) == 2


def test_encoded_table_frame_includes_fields_needed_for_selection(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "PDMX"
    encoded_directory = dataset_dir / "encoded" / "abc"
    encoded_directory.mkdir(parents=True)
    _write_csv(dataset_dir / "parsed.csv", ParsedManifestField, [{ParsedManifestField.SOURCE_ID: "a"}])
    _write_csv(
        encoded_directory / "encoded.csv",
        EncodedManifestField,
        [
            {
                EncodedManifestField.SEGMENT_ID: "a-0",
                EncodedManifestField.SOURCE_ID: "a",
                EncodedManifestField.PARSED_PATH: "parsed/a/a.json",
            }
        ],
    )

    stats = load_dataset_statistics(dataset_dir, encoded_directory)
    assert stats.encoded is not None
    frame = encoded_table_frame(stats.encoded)

    assert "source_id" in frame.columns
    assert "parsed_path" in frame.columns


def test_selected_table_row_normalizes_table_values() -> None:
    class FakeTable:
        value = [{"segment_id": "a", "encoded_line": 0}]

    assert selected_table_row(FakeTable()) == {"segment_id": "a", "encoded_line": 0}
    assert selected_table_row(object()) is None


def _write_csv(path: Path, field_type, rows: list[dict[object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.value for field in field_type]
    normalized_rows = []
    for row in rows:
        normalized_rows.append({field.value: row.get(field, "") for field in field_type})

    lines = [",".join(fieldnames)]
    for row in normalized_rows:
        lines.append(",".join(str(row[field]) for field in fieldnames))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
