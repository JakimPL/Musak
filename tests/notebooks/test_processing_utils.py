import sys
import warnings
from fractions import Fraction
from pathlib import Path

from musak_model.processing.manifest import parsed_success_row, write_parsed_manifest
from notebooks.utils.processing import (
    encoded_segments_result,
    parsed_score_manifest_diagnostics,
    process_score_safely,
)
from tests.data.fixtures import bar, note_event, parsed_score


def _score():
    return parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
    )


def test_process_score_safely_captures_parse_diagnostics_without_console_noise(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source_path = tmp_path / "piece.mxl"
    source_path.write_text("score", encoding="utf-8")

    def noisy_parse(path: Path):
        warnings.warn("musicxml warning", UserWarning, stacklevel=1)
        print("stderr diagnostic", file=sys.stderr)
        return _score()

    monkeypatch.setattr("notebooks.utils.processing.parse_score", noisy_parse)
    monkeypatch.setattr("notebooks.utils.processing.segment_parsed_score", lambda *args, **kwargs: [])

    result = process_score_safely(source_path, window_bars=1, stride_bars=1)
    captured = capsys.readouterr()

    assert result.succeeded
    assert "UserWarning: musicxml warning" in result.parse_diagnostics
    assert "stderr diagnostic" in result.parse_diagnostics
    assert "musicxml warning" not in captured.err
    assert "stderr diagnostic" not in captured.err


def test_process_score_safely_preserves_parse_diagnostics_on_parse_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "piece.mxl"
    source_path.write_text("score", encoding="utf-8")

    def noisy_parse_error(path: Path):
        warnings.warn("before failure", UserWarning, stacklevel=1)
        raise ValueError("bad score")

    monkeypatch.setattr("notebooks.utils.processing.parse_score", noisy_parse_error)

    result = process_score_safely(source_path, window_bars=1, stride_bars=1)

    assert not result.succeeded
    assert result.error_type == "ValueError"
    assert "UserWarning: before failure" in result.parse_diagnostics


def test_parsed_score_manifest_diagnostics_reads_nearest_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "mxl" / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score", encoding="utf-8")
    processed_root = tmp_path / "processed" / "PDMX"
    parsed_path = processed_root / "parsed" / "a" / "abc.json"
    parsed_path.parent.mkdir(parents=True)
    parsed_path.write_text("{}", encoding="utf-8")
    write_parsed_manifest(
        [
            parsed_success_row(
                source_id_value="abc",
                source_path=source_path,
                dataset_root=dataset_root,
                title="Piece",
                parsed_path=parsed_path,
                processed_root=processed_root,
                score=_score(),
                parse_diagnostics="MusicXMLWarning: bad style",
            )
        ],
        processed_root / "parsed.csv",
    )

    assert parsed_score_manifest_diagnostics(parsed_path) == "MusicXMLWarning: bad style"


def test_parsed_score_manifest_diagnostics_returns_empty_string_without_manifest(tmp_path: Path) -> None:
    assert parsed_score_manifest_diagnostics(tmp_path / "parsed" / "a" / "abc.json") == ""


def test_encoded_segments_result_has_no_parse_diagnostics(tmp_path: Path) -> None:
    assert encoded_segments_result(tmp_path / "data.jsonl", segments=[]).parse_diagnostics == ""
