from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

import pytest

from musak_model.data.config import SegmentationConfig
from musak_model.processing import dataset as dataset_module
from musak_model.processing.dataset import process_dataset
from musak_model.processing.ids import source_id
from musak_model.processing.io import load_encoded_jsonl
from musak_model.processing.manifest import (
    EncodedManifestField,
    ParsedManifestField,
    ParsedManifestStatus,
    read_encoded_manifest,
    read_parsed_manifest,
)
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from tests.data.fixtures import bar, note_event, parsed_score


def _score():
    return parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
    )


def _segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(window_bars=1, stride_bars=1)


def test_process_dataset_writes_parsed_and_encoded_artifacts(
    tmp_path: Path,
    monkeypatch,
    tokenization_config: TokenizationConfig,
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
        segmentation=_segmentation_config(),
        tokenization_config=tokenization_config,
        stage="all",
        overwrite=True,
    )

    parsed_rows = read_parsed_manifest(result.parsed_manifest_path)
    encoded_rows = read_encoded_manifest(result.encoded_manifest_path or Path())

    assert result.parsed_count == 1
    assert result.encoded_count == 1
    assert result.error_count == 0
    assert result.parsed_manifest_path.parent == processed_root / "PDMX"
    assert result.encoded_manifest_path is not None
    assert result.tokenizer_snapshot_path is not None and result.tokenizer_snapshot_path.exists()
    assert parsed_rows[0][ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value
    assert parsed_rows[0][ParsedManifestField.TITLE] == "Piece"
    expected_source_id = source_id(source_path, dataset_root=dataset_root)
    assert parsed_rows[0][ParsedManifestField.PARSED_PATH] == (
        f"parsed/{expected_source_id[0]}/{expected_source_id}.json"
    )
    assert Path(processed_root / "PDMX" / parsed_rows[0][ParsedManifestField.PARSED_PATH]).exists()
    assert encoded_rows[0][EncodedManifestField.ELIGIBLE_FOR_TRAINING] == "True"
    assert encoded_rows[0][EncodedManifestField.ENCODED_LINE] == "0"
    assert load_encoded_jsonl(result.encoded_manifest_path.parent / "data-00000.jsonl")[0].source_file == Path(
        "mxl/piece.mxl"
    )


@pytest.mark.parametrize(
    ("exception", "error_type"),
    [
        (ValueError("bad score"), "ValueError"),
        (OverflowError("cannot convert float infinity to integer"), "OverflowError"),
    ],
)
def test_process_dataset_records_parse_errors(
    tmp_path: Path,
    monkeypatch,
    tokenization_config: TokenizationConfig,
    exception: Exception,
    error_type: str,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")

    def fail_parse(path: Path):
        raise exception

    monkeypatch.setattr("musak_model.processing.dataset.parse_score", fail_parse)
    monkeypatch.setattr("musak_model.processing.dataset._score_title", lambda path: "")

    result = process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        segmentation=_segmentation_config(),
        tokenization_config=tokenization_config,
        stage="parsed",
        overwrite=True,
    )

    parsed_rows = read_parsed_manifest(result.parsed_manifest_path)

    assert result.parsed_count == 0
    assert result.error_count == 1
    assert parsed_rows[0][ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value
    assert parsed_rows[0][ParsedManifestField.ERROR_TYPE] == error_type


def test_process_dataset_rejects_invalid_worker_count(tmp_path: Path, tokenization_config: TokenizationConfig) -> None:
    with pytest.raises(ValueError, match="workers"):
        process_dataset(
            tmp_path,
            processed_root=tmp_path / "processed",
            segmentation=_segmentation_config(),
            tokenization_config=tokenization_config,
            stage="parsed",
            workers=0,
        )


def test_process_dataset_parallel_parse_keeps_manifest_order(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    dataset_root.mkdir()
    for filename in ("b.musicxml", "a.musicxml", "c.musicxml"):
        (dataset_root / filename).write_text("<score-partwise>", encoding="utf-8")

    result = process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        segmentation=_segmentation_config(),
        tokenization_config=tokenization_config,
        stage="parsed",
        workers=2,
        overwrite=True,
        show_progress=False,
    )

    parsed_rows = read_parsed_manifest(result.parsed_manifest_path)

    assert result.parsed_count == 0
    assert result.error_count == 3
    assert [row[ParsedManifestField.SOURCE_FILENAME] for row in parsed_rows] == [
        "a.musicxml",
        "b.musicxml",
        "c.musicxml",
    ]


def test_process_dataset_rebuilds_incomplete_encoded_outputs(
    tmp_path: Path,
    monkeypatch,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")
    processed_root = tmp_path / "processed"
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    encoded_jsonl_path = paths.encoded_jsonl_path(snapshot.tokenizer_hash)
    encoded_jsonl_path.parent.mkdir(parents=True)
    encoded_jsonl_path.write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr("musak_model.processing.dataset.parse_score", lambda path: _score())
    monkeypatch.setattr("musak_model.processing.dataset._score_title", lambda path: "Piece")

    result = process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation=_segmentation_config(),
        tokenization_config=tokenization_config,
        stage="all",
        overwrite=False,
        workers=1,
    )

    assert result.encoded_count == 1
    assert encoded_jsonl_path.read_text(encoding="utf-8") != "stale\n"
    assert len(load_encoded_jsonl(encoded_jsonl_path)) == 1


def test_score_title_returns_musicxml_movement_title(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.musicxml"
    source_path.write_text(
        """
        <score-partwise>
          <movement-title>Prelude</movement-title>
        </score-partwise>
        """,
        encoding="utf-8",
    )

    assert dataset_module._score_title(source_path) == "Prelude"


def test_score_title_returns_musicxml_work_title(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.musicxml"
    source_path.write_text(
        """
        <score-partwise>
          <work>
            <work-title>Notebook Sketch</work-title>
          </work>
        </score-partwise>
        """,
        encoding="utf-8",
    )

    assert dataset_module._score_title(source_path) == "Notebook Sketch"


def test_score_title_returns_compressed_musicxml_title(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.mxl"
    with ZipFile(source_path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            """
            <container>
              <rootfiles>
                <rootfile full-path="score.musicxml" />
              </rootfiles>
            </container>
            """,
        )
        archive.writestr(
            "score.musicxml",
            """
            <score-partwise>
              <movement-title>Compressed Piece</movement-title>
            </score-partwise>
            """,
        )

    assert dataset_module._score_title(source_path) == "Compressed Piece"


def test_score_title_returns_empty_for_malformed_musicxml(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.musicxml"
    source_path.write_text("<score-partwise>")

    assert dataset_module._score_title(source_path) == ""
