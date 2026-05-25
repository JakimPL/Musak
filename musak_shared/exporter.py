import base64
import io
import pathlib
import subprocess
import tempfile
from typing import Final

import mido

SOUNDFONT: Final[str] = "st_concert.sf2"
GAIN: Final[float] = 3.5


class AudioExportError(RuntimeError):
    pass


def midi_to_audio(midi_file: mido.MidiFile) -> str:
    buffer = io.BytesIO()
    midi_file.save(file=buffer)
    return Exporter().export_audio(buffer.getvalue())


class Exporter:
    def __init__(
        self,
        *,
        sf2: str = SOUNDFONT,
        gain: float = GAIN,
        soundfont_directory: pathlib.Path | None = None,
    ) -> None:
        root = pathlib.Path.cwd() if soundfont_directory is None else soundfont_directory
        self.soundfont_path: pathlib.Path = root / "soundfont" / sf2 if soundfont_directory is None else root / sf2
        self.gain: float = gain

    def export_audio(self, midi_data: bytes) -> str:
        if not self.soundfont_path.exists():
            raise AudioExportError(f"soundfont not found: {self.soundfont_path}")

        with tempfile.TemporaryDirectory() as tmp:
            temp_directory = pathlib.Path(tmp)
            midi_path = temp_directory / "audio.mid"
            wav_path = temp_directory / "audio.wav"
            mp3_path = temp_directory / "audio.mp3"
            midi_path.write_bytes(midi_data)
            self._to_audio(midi_path, wav_path)
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(wav_path), str(mp3_path)],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exception:
                raise AudioExportError(f"ffmpeg failed with exit code {exception.returncode}") from exception
            except FileNotFoundError as exception:
                raise AudioExportError("ffmpeg executable not found") from exception

            return "data:audio/mpeg;base64," + base64.b64encode(mp3_path.read_bytes()).decode()

    def _to_audio(self, midi_file: pathlib.Path, out_file: pathlib.Path) -> None:
        try:
            subprocess.run(
                [
                    "fluidsynth",
                    "-g",
                    str(self.gain),
                    "-T",
                    "wav",
                    "-F",
                    str(out_file),
                    "-ni",
                    str(self.soundfont_path),
                    str(midi_file),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exception:
            raise AudioExportError(f"fluidsynth failed with exit code {exception.returncode}") from exception
        except FileNotFoundError as exception:
            raise AudioExportError("fluidsynth executable not found") from exception
