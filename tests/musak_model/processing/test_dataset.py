import sys
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest

from musak_model.data.config import DataProcessingConfig, SegmentationConfig, SegmentationMode
from musak_model.data.schema import SegmentIneligibilityReason
from musak_model.processing import parse as parse_module
from musak_model.processing import tokenize as tokenize_module
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
from musak_model.processing.profiler import ProcessingProfiler
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from tests.musak_model.data.fixtures import bar, note_event, parsed_score, rest_event


def _score():
    return parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
    )


def _score_with_interior_silent_bar():
    right_note_bar = bar([note_event(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(0))])
    left_note_bar = bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))])
    silent_bar = bar([rest_event(duration=Fraction(1, 1), beat_offset=Fraction(0))])
    return parsed_score(
        right_hand_bars=[right_note_bar, silent_bar, right_note_bar],
        left_hand_bars=[left_note_bar, silent_bar, left_note_bar],
    )


def _score_with_three_active_bars():
    right_note_bar = bar([note_event(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(0))])
    left_note_bar = bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))])
    return parsed_score(
        right_hand_bars=[right_note_bar, right_note_bar, right_note_bar],
        left_hand_bars=[left_note_bar, left_note_bar, left_note_bar],
    )


def _segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(window_bars=1, stride_bars=1)


def test_process_dataset_writes_parsed_and_encoded_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "mxl" / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")
    processed_root = tmp_path / "processed"

    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: _score())
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "Piece")

    result = process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="process",
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
    assert int(encoded_rows[0][EncodedManifestField.TOKEN_COUNT]) > 0
    assert load_encoded_jsonl(result.encoded_manifest_path.parent / "data-00000.jsonl")[0].source_file == Path(
        "mxl/piece.mxl"
    )


def test_process_dataset_marks_segments_with_any_silent_bar_ineligible_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")

    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: _score_with_interior_silent_bar())
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "Piece")

    result = process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        segmentation_config=SegmentationConfig(window_bars=3, stride_bars=1),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="process",
        overwrite=True,
    )

    encoded_rows = read_encoded_manifest(result.encoded_manifest_path or Path())

    assert result.encoded_count == 0
    assert encoded_rows[0][EncodedManifestField.ELIGIBLE_FOR_TRAINING] == "False"
    assert encoded_rows[0][EncodedManifestField.ENCODED_LINE] == ""
    assert encoded_rows[0][EncodedManifestField.INELIGIBILITY_REASONS] == SegmentIneligibilityReason.SILENT_BAR.value
    assert encoded_rows[0][EncodedManifestField.SILENT_BAR_COUNT] == "1"


def test_process_dataset_can_keep_segments_with_silent_bars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")

    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: _score_with_interior_silent_bar())
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "Piece")

    result = process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        segmentation_config=SegmentationConfig(window_bars=3, stride_bars=1),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=False),
        stage="process",
        overwrite=True,
    )

    encoded_rows = read_encoded_manifest(result.encoded_manifest_path or Path())

    assert result.encoded_count == 1
    assert encoded_rows[0][EncodedManifestField.ELIGIBLE_FOR_TRAINING] == "True"
    assert encoded_rows[0][EncodedManifestField.INELIGIBILITY_REASONS] == ""
    assert encoded_rows[0][EncodedManifestField.SILENT_BAR_COUNT] == "1"


def test_process_dataset_whole_file_segmentation_records_full_bar_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "exercises"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")

    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: _score_with_three_active_bars())
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "Piece")

    result = process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1, mode=SegmentationMode.WHOLE_FILE),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="process",
        overwrite=True,
    )

    encoded_rows = read_encoded_manifest(result.encoded_manifest_path or Path())
    encoded_sample = load_encoded_jsonl(result.encoded_manifest_path.parent / "data-00000.jsonl")[0]

    assert result.encoded_count == 1
    assert len(encoded_rows) == 1
    assert encoded_rows[0][EncodedManifestField.BAR_COUNT] == "3"
    assert encoded_sample.metadata.bar_count == 3


def test_process_dataset_records_processing_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "exercises"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")
    profiler = ProcessingProfiler(output_dir=tmp_path / "profile", use_torch_profiler_labels=False)

    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: _score())
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "Piece")

    process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="process",
        overwrite=True,
        profiler=profiler,
    )

    stages = {record.stage for record in profiler.records}

    assert "process_parsed_scores" in stages
    assert "scale_match" in stages
    assert "encode_segment" in stages
    assert "append_encoded_manifest" in stages


def test_process_dataset_warns_about_unspecified_difficulty_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "exercises"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")

    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: _score())
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "Piece")

    with caplog.at_level("WARNING", logger="musak_model.processing.tokenize"):
        process_dataset(
            dataset_root,
            processed_root=tmp_path / "processed",
            segmentation_config=_segmentation_config(),
            tokenization_config=tokenization_config,
            data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
            stage="process",
            difficulty_labels={"other.mxl": 2},
            overwrite=True,
        )

    assert "Difficulty labels: labeled=0 explicit_unlabeled=0 unspecified=1" in caplog.text


@pytest.mark.parametrize(
    ("exception", "error_type"),
    [
        (ValueError("bad score"), "ValueError"),
        (OverflowError("cannot convert float infinity to integer"), "OverflowError"),
    ],
)
def test_process_dataset_records_parse_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr("musak_model.processing.parse.parse_score", fail_parse)
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "")

    result = process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="parse",
        overwrite=True,
    )

    parsed_rows = read_parsed_manifest(result.parsed_manifest_path)

    assert result.parsed_count == 0
    assert result.error_count == 1
    assert parsed_rows[0][ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value
    assert parsed_rows[0][ParsedManifestField.ERROR_TYPE] == error_type


def test_process_dataset_records_parse_diagnostics_without_console_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")

    def noisy_parse(path: Path):
        warnings.warn("musicxml warning", UserWarning, stacklevel=1)
        print("stderr diagnostic", file=sys.stderr)
        return _score()

    monkeypatch.setattr("musak_model.processing.parse.parse_score", noisy_parse)
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "")

    result = process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="parse",
        overwrite=True,
    )

    parsed_rows = read_parsed_manifest(result.parsed_manifest_path)
    captured = capsys.readouterr()

    assert parsed_rows[0][ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value
    assert "UserWarning: musicxml warning" in parsed_rows[0][ParsedManifestField.PARSE_DIAGNOSTICS]
    assert "stderr diagnostic" in parsed_rows[0][ParsedManifestField.PARSE_DIAGNOSTICS]
    assert "musicxml warning" not in captured.err
    assert "stderr diagnostic" not in captured.err


def test_process_dataset_records_parse_diagnostics_on_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")

    def noisy_parse_error(path: Path):
        warnings.warn("before failure", UserWarning, stacklevel=1)
        raise ValueError("bad score")

    monkeypatch.setattr("musak_model.processing.parse.parse_score", noisy_parse_error)
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "")

    result = process_dataset(
        dataset_root,
        processed_root=tmp_path / "processed",
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="parse",
        overwrite=True,
    )

    parsed_rows = read_parsed_manifest(result.parsed_manifest_path)

    assert parsed_rows[0][ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value
    assert parsed_rows[0][ParsedManifestField.ERROR_TYPE] == "ValueError"
    assert "UserWarning: before failure" in parsed_rows[0][ParsedManifestField.PARSE_DIAGNOSTICS]


def test_process_dataset_rejects_invalid_worker_count(tmp_path: Path, tokenization_config: TokenizationConfig) -> None:
    with pytest.raises(ValueError, match="workers"):
        process_dataset(
            tmp_path,
            processed_root=tmp_path / "processed",
            segmentation_config=_segmentation_config(),
            tokenization_config=tokenization_config,
            data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
            stage="parse",
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
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="parse",
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


def test_process_dataset_reuses_error_rows_from_parsed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")
    processed_root = tmp_path / "processed"

    monkeypatch.setattr(
        "musak_model.processing.parse.parse_score", lambda path: (_ for _ in ()).throw(ValueError("bad score"))
    )
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "")
    process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="parse",
        overwrite=True,
    )

    def fail_if_reparsed(path: Path):
        raise AssertionError("rejected source should be skipped")

    monkeypatch.setattr("musak_model.processing.parse.parse_score", fail_if_reparsed)

    with caplog.at_level("INFO", logger="musak_model.processing.parse"):
        result = process_dataset(
            dataset_root,
            processed_root=processed_root,
            segmentation_config=_segmentation_config(),
            tokenization_config=tokenization_config,
            data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
            stage="parse",
            overwrite=False,
        )

    assert result.parsed_count == 0
    assert result.error_count == 1
    assert "Reusing 1 parsed manifest row(s): 0 success(es), 1 error(s)" in caplog.text
    assert "Parsing 0/1 source file(s)" in caplog.text


def test_process_dataset_reuses_success_rows_from_parsed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")
    processed_root = tmp_path / "processed"

    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: _score())
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "Piece")
    process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="parse",
        overwrite=True,
    )

    def fail_if_reparsed(path: Path):
        raise AssertionError("parsed source should be loaded from parsed JSON")

    monkeypatch.setattr("musak_model.processing.parse.parse_score", fail_if_reparsed)

    result = process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="parse",
        overwrite=False,
    )

    assert result.parsed_count == 1
    assert result.error_count == 0


def test_process_dataset_writes_partial_parsed_manifest_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    dataset_root.mkdir()
    first_path = dataset_root / "a.musicxml"
    second_path = dataset_root / "b.musicxml"
    first_path.write_text("score", encoding="utf-8")
    second_path.write_text("score", encoding="utf-8")
    processed_root = tmp_path / "processed"
    call_count = 0

    def interrupt_after_first(path: Path):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _score()
        raise KeyboardInterrupt

    monkeypatch.setattr("musak_model.processing.parse.parse_score", interrupt_after_first)
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "")

    with pytest.raises(KeyboardInterrupt):
        process_dataset(
            dataset_root,
            processed_root=processed_root,
            segmentation_config=_segmentation_config(),
            tokenization_config=tokenization_config,
            data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
            stage="parse",
            overwrite=True,
            workers=1,
        )

    parsed_rows = read_parsed_manifest(processed_root / "PDMX" / "parsed.csv")

    assert len(parsed_rows) == 1
    assert parsed_rows[0][ParsedManifestField.SOURCE_FILENAME] == "a.musicxml"
    assert parsed_rows[0][ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value


def test_process_dataset_rebuilds_incomplete_encoded_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: _score())
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "Piece")

    result = process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="process",
        overwrite=False,
        workers=1,
    )

    assert result.encoded_count == 1
    assert encoded_jsonl_path.read_text(encoding="utf-8") != "stale\n"
    assert len(load_encoded_jsonl(encoded_jsonl_path)) == 1


def test_process_dataset_resumes_interrupted_tokenization_from_completed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    dataset_root.mkdir()
    (dataset_root / "a.mxl").write_text("score")
    (dataset_root / "b.mxl").write_text("score")
    processed_root = tmp_path / "processed"

    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: _score())
    monkeypatch.setattr("musak_model.processing.parse._score_title", lambda path: "Piece")
    process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="parse",
        overwrite=True,
    )

    original_tokenize_source = tokenize_module._tokenize_source
    call_count = 0

    def interrupt_after_second_source(*args: Any, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        result = original_tokenize_source(*args, **kwargs)
        if call_count == 2:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(tokenize_module, "_tokenize_source", interrupt_after_second_source)
    with pytest.raises(KeyboardInterrupt):
        process_dataset(
            dataset_root,
            processed_root=processed_root,
            segmentation_config=_segmentation_config(),
            tokenization_config=tokenization_config,
            data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
            stage="tokenize",
            overwrite=False,
        )

    monkeypatch.setattr(tokenize_module, "_tokenize_source", original_tokenize_source)
    result = process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=_segmentation_config(),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="tokenize",
        overwrite=False,
    )

    assert result.encoded_count == 2
    assert len(read_encoded_manifest(result.encoded_manifest_path or Path())) == 2
    assert len(load_encoded_jsonl(result.encoded_manifest_path.parent / "data-00000.jsonl")) == 2


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

    assert parse_module._score_title(source_path) == "Prelude"


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

    assert parse_module._score_title(source_path) == "Notebook Sketch"


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

    assert parse_module._score_title(source_path) == "Compressed Piece"


def test_score_title_returns_empty_for_malformed_musicxml(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.musicxml"
    source_path.write_text("<score-partwise>")

    assert parse_module._score_title(source_path) == ""
