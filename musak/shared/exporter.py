import pathlib
import subprocess
from typing import Final

SOUNDFONT: Final[str] = "st_concert.sf2"
AUDIO_FORMAT: Final[str] = "wav"
CONVERT_TO_MP3: Final[bool] = True
GAIN: Final[float] = 3.5


class Exporter:
    def __init__(
        self,
        name: str,
        *,
        sf2: str = SOUNDFONT,
        audio_format: str = AUDIO_FORMAT,
        convert_to_mp3: bool = CONVERT_TO_MP3,
        gain: float = GAIN,
    ) -> None:
        self.name: str = name
        self.soundfont_path: pathlib.Path = pathlib.Path.cwd() / "soundfont" / sf2
        self.audio_format: str = audio_format
        self.convert_to_mp3: bool = convert_to_mp3
        self.gain: float = gain

    def export_audio(
        self,
        midi_path: pathlib.Path,
        audio_path: pathlib.Path,
    ) -> pathlib.Path:
        self._to_audio(
            self.soundfont_path,
            midi_path,
            audio_path,
            out_type=self.audio_format,
        )
        if self.convert_to_mp3:
            mp3_path = audio_path.with_suffix(".mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), str(mp3_path)],
                check=True,
                capture_output=True,
            )
            return mp3_path

        return audio_path

    def _to_audio(
        self,
        sf2: pathlib.Path,
        midi_file: pathlib.Path,
        out_file: pathlib.Path,
        *,
        out_type: str = "wav",
    ) -> None:
        subprocess.call(
            [
                "fluidsynth",
                "-g",
                str(self.gain),
                "-T",
                out_type,
                "-F",
                str(out_file),
                "-ni",
                str(sf2),
                str(midi_file),
            ]
        )
