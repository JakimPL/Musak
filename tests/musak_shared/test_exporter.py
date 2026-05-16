import subprocess
from unittest.mock import patch

import pytest

from musak_shared.exporter import AudioExportError, Exporter


def _make_exporter(tmp_path) -> Exporter:
    (tmp_path / "st_concert.sf2").write_bytes(b"soundfont")
    return Exporter(soundfont_dir=tmp_path)


def _fluidsynth_error() -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(returncode=1, cmd=["fluidsynth"])


def _ffmpeg_error() -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"])


def test_fluidsynth_failure_raises_audio_export_error(tmp_path) -> None:
    with patch("musak_shared.exporter.subprocess.run", side_effect=_fluidsynth_error()):
        with pytest.raises(AudioExportError, match="fluidsynth failed with exit code 1"):
            _make_exporter(tmp_path).export_audio(b"")


def test_ffmpeg_failure_raises_audio_export_error(tmp_path) -> None:
    def _side_effect(cmd: list[str], **kwargs: object) -> None:
        if cmd[0] == "fluidsynth":
            return None
        raise _ffmpeg_error()

    with patch("musak_shared.exporter.subprocess.run", side_effect=_side_effect):
        with pytest.raises(AudioExportError, match="ffmpeg failed with exit code 1"):
            _make_exporter(tmp_path).export_audio(b"")


def test_missing_soundfont_raises_audio_export_error(tmp_path) -> None:
    with pytest.raises(AudioExportError, match="soundfont not found"):
        Exporter(soundfont_dir=tmp_path).export_audio(b"")


def test_audio_export_error_is_runtime_error() -> None:
    assert issubclass(AudioExportError, RuntimeError)
