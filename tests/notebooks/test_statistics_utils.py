from pathlib import Path

import pytest

from musak_model.processing.manifest import EncodedManifestField, ParsedManifestField
from notebooks.utils.statistics import (
    COUNT_COLUMN,
    VALUE_COLUMN,
    DatasetStatistics,
    eligibility_distribution,
    ineligibility_reason_distribution,
    load_dataset_statistics,
    overview_rows,
    read_encoded_manifest_frame,
    reason_by_column,
    token_summary_rows,
)


def test_load_dataset_statistics_requires_current_encoded_manifest_columns(tmp_path: Path) -> None:
    encoded_path = tmp_path / "encoded.csv"
    encoded_path.write_text("segment_id,source_id\nabc,source\n", encoding="utf-8")

    with pytest.raises(ValueError, match="token_count"):
        read_encoded_manifest_frame(encoded_path)


def test_dataset_overview_counts_parsed_and_encoded_rows(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "PDMX"
    encoded_dir = dataset_dir / "encoded" / "abc"
    encoded_dir.mkdir(parents=True)
    _write_csv(
        dataset_dir / "parsed.csv",
        ParsedManifestField,
        [
            {ParsedManifestField.SOURCE_ID: "a", ParsedManifestField.STATUS: "success"},
            {ParsedManifestField.SOURCE_ID: "b", ParsedManifestField.STATUS: "error"},
        ],
    )
    _write_csv(
        encoded_dir / "encoded.csv",
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

    stats = load_dataset_statistics(dataset_dir, encoded_dir)

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
    encoded_dir = dataset_dir / "encoded" / "abc"
    encoded_dir.mkdir(parents=True)
    _write_csv(dataset_dir / "parsed.csv", ParsedManifestField, [{ParsedManifestField.SOURCE_ID: "a"}])
    _write_csv(
        encoded_dir / "encoded.csv",
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
    stats = load_dataset_statistics(dataset_dir, encoded_dir)
    assert stats.encoded is not None

    eligibility = eligibility_distribution(stats.encoded)
    reasons = ineligibility_reason_distribution(stats.encoded)
    reason_time = reason_by_column(stats.encoded, EncodedManifestField.TIME_SIGNATURE)
    token_rows = token_summary_rows(stats.encoded)

    assert eligibility.set_index(VALUE_COLUMN).loc["ineligible", COUNT_COLUMN] == 2
    assert reasons.set_index(VALUE_COLUMN).loc["quantization_error", COUNT_COLUMN] == 2
    assert len(reason_time) == 3
    assert token_rows[0] == {"Metric": "min", "Value": "10"}
    assert token_rows[2] == {"Metric": "median", "Value": "20"}
    assert token_rows[-1] == {"Metric": "max", "Value": "30"}


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
