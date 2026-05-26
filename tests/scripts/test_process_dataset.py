import logging
from pathlib import Path

import pytest

from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.processing.dataset import ProcessDatasetResult
from scripts import process_dataset as process_dataset_script


def test_source_files_for_parse_stage_exits_when_no_musicxml(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger=process_dataset_script._LOGGER.name)

    with pytest.raises(SystemExit) as exit_info:
        process_dataset_script._source_files_for_stage(tmp_path, stage="parse")

    assert exit_info.value.code == 1
    assert "No MusicXML files found" in caplog.text
    assert "no parsed manifest was created" in caplog.text


def test_source_files_for_tokenize_stage_does_not_scan_raw_musicxml(tmp_path: Path) -> None:
    assert process_dataset_script._source_files_for_stage(tmp_path, stage="tokenize") == []


def test_source_files_for_parse_stage_returns_musicxml_files(tmp_path: Path) -> None:
    source_file = tmp_path / "score.mxl"
    source_file.write_text("", encoding="utf-8")

    assert process_dataset_script._source_files_for_stage(tmp_path, stage="parse") == [source_file]


def test_missing_processing_input_log_mentions_parse_stage(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger=process_dataset_script._LOGGER.name)

    process_dataset_script._log_processing_file_not_found(
        FileNotFoundError("parsed manifest does not exist: processed/0/parsed.csv"),
        data_directory=Path("data/PDMX/0"),
        processed_directory=Path("processed"),
    )

    assert "parsed manifest does not exist: processed/0/parsed.csv" in caplog.text
    assert "run the parse stage first with the same --data-dir" in caplog.text
    assert "processed/0" in caplog.text


@pytest.mark.parametrize(
    ("stage", "expected_name"),
    [
        ("parse", "parse"),
        ("tokenize", "tokenize"),
        ("process", "process"),
    ],
)
def test_profile_output_dir_defaults_to_stage_specific_directory(stage: str, expected_name: str) -> None:
    assert process_dataset_script._profile_output_dir(stage, configured=None).name == expected_name


def test_profile_output_dir_uses_explicit_directory(tmp_path: Path) -> None:
    configured = tmp_path / "profile"

    assert process_dataset_script._profile_output_dir("parse", configured=configured) == configured


def test_extract_process_figure_artifacts_skips_parse_result(tmp_path: Path) -> None:
    result = ProcessDatasetResult(
        parsed_manifest_path=tmp_path / "parsed.csv",
        encoded_manifest_path=None,
        tokenizer_snapshot_path=None,
        parsed_count=1,
        encoded_count=0,
        error_count=0,
        scale_matcher_config=ScaleMatcherConfig(
            support_score_margin=0.08,
            selection_score_margin=0.03,
            maximum_unexplained_weight_fraction=0.10,
            maximum_explanation_pitch_class_count=9,
        ),
    )

    assert (
        process_dataset_script.extract_process_figure_artifacts(
            result=result,
            analysis_config_path=tmp_path / "n_grams.yml",
            output_path=None,
            skip_figure_analysis=False,
            show_progress=False,
        )
        is None
    )


def test_extract_process_figure_artifacts_respects_skip_flag(tmp_path: Path) -> None:
    result = ProcessDatasetResult(
        parsed_manifest_path=tmp_path / "parsed.csv",
        encoded_manifest_path=tmp_path / "encoded" / "abc" / "encoded.csv",
        tokenizer_snapshot_path=tmp_path / "encoded" / "abc" / "tokenizer.json",
        parsed_count=1,
        encoded_count=1,
        error_count=0,
        scale_matcher_config=ScaleMatcherConfig(
            support_score_margin=0.08,
            selection_score_margin=0.03,
            maximum_unexplained_weight_fraction=0.10,
            maximum_explanation_pitch_class_count=9,
        ),
    )

    assert (
        process_dataset_script.extract_process_figure_artifacts(
            result=result,
            analysis_config_path=tmp_path / "n_grams.yml",
            output_path=None,
            skip_figure_analysis=True,
            show_progress=False,
        )
        is None
    )
