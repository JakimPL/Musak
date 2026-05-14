from fractions import Fraction
from pathlib import Path

from music21.metadata import Metadata
from music21.stream.base import Score

from musak_model.data.config import SegmentationConfig
from musak_model.processing import dataset as dataset_module
from musak_model.processing.dataset import process_dataset
from musak_model.processing.manifest import (
    EncodedManifestField,
    ParsedManifestField,
    ParsedManifestStatus,
    read_encoded_manifest,
    read_parsed_manifest,
)
from musak_model.tokens.config import TokenizationConfig
from tests.data.fixtures import bar, note_event, parsed_score


def _score():
    return parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
    )


def test_process_dataset_writes_parsed_and_encoded_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "mxl" / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")
    processed_root = tmp_path / "processed"

    monkeypatch.setattr("musak_model.processing.dataset.parse_score", lambda path: _score())
    monkeypatch.setattr("musak_model.processing.dataset._score_title", lambda path: "Piece")

    result = process_dataset(
        dataset_root,
        processed_root=processed_root,
        dataset_name="PDMX",
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1),
        stage="all",
        overwrite=True,
    )

    parsed_rows = read_parsed_manifest(result.parsed_manifest_path)
    encoded_rows = read_encoded_manifest(result.encoded_manifest_path or Path())

    assert result.parsed_count == 1
    assert result.encoded_count == 1
    assert result.error_count == 0
    assert result.tokenizer_snapshot_path is not None and result.tokenizer_snapshot_path.exists()
    assert parsed_rows[0][ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value
    assert parsed_rows[0][ParsedManifestField.TITLE] == "Piece"
    assert Path(processed_root / "PDMX" / parsed_rows[0][ParsedManifestField.PARSED_PATH]).exists()
    assert encoded_rows[0][EncodedManifestField.ELIGIBLE_FOR_TRAINING] == "True"
    assert encoded_rows[0][EncodedManifestField.ENCODED_LINE] == "0"


def test_process_dataset_records_parse_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")

    def fail_parse(path: Path):
        raise ValueError("bad score")

    monkeypatch.setattr("musak_model.processing.dataset.parse_score", fail_parse)
    monkeypatch.setattr("musak_model.processing.dataset._score_title", lambda path: "")

    result = process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        dataset_name="PDMX",
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1),
        stage="parsed",
        overwrite=True,
    )

    parsed_rows = read_parsed_manifest(result.parsed_manifest_path)

    assert result.parsed_count == 0
    assert result.error_count == 1
    assert parsed_rows[0][ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value
    assert parsed_rows[0][ParsedManifestField.ERROR_TYPE] == "ValueError"


def test_score_title_returns_metadata_title(tmp_path: Path, monkeypatch) -> None:
    score = Score()
    score.metadata = Metadata()
    score.metadata.title = "Prelude"
    source_path = tmp_path / "piece.musicxml"

    monkeypatch.setattr(dataset_module.converter, "parse", lambda path: score)

    assert dataset_module._score_title(source_path) == "Prelude"


def test_score_title_returns_empty_for_malformed_musicxml(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.musicxml"
    source_path.write_text("<score-partwise>")

    assert dataset_module._score_title(source_path) == ""
