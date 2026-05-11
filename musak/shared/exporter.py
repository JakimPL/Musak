import pathlib
import subprocess
from typing import Final

import abjad
import pydub

SOUNDFONT: Final[str] = "st_concert.sf2"
AUDIO_FORMAT: Final[str] = "wav"
CONVERT_TO_MP3: Final[bool] = True
IGNORE_MIDI: Final[bool] = False
IGNORE_AUDIO: Final[bool] = False
IGNORE_SCORE: Final[bool] = False
GAIN: Final[float] = 3.5


class Exporter:
    def __init__(
        self,
        name: str,
        *,
        sf2: str = SOUNDFONT,
        ignore_midi: bool = IGNORE_MIDI,
        ignore_audio: bool = IGNORE_AUDIO,
        ignore_score: bool = IGNORE_SCORE,
        audio_format: str = AUDIO_FORMAT,
        convert_to_mp3: bool = CONVERT_TO_MP3,
        gain: float = GAIN,
    ) -> None:
        self.name: str = name
        self.soundfont_path: pathlib.Path = pathlib.Path.cwd() / "soundfont" / sf2
        self.audio_format: str = audio_format
        self.convert_to_mp3: bool = convert_to_mp3
        self.gain: float = gain
        self.ignore_midi: bool = ignore_midi
        self.ignore_audio: bool = ignore_audio
        self.ignore_score: bool = ignore_score

    @staticmethod
    def prepare_ly_file(score: abjad.Score) -> abjad.LilyPondFile:
        score_block = abjad.Block("score", items=[score])
        score_block.items.append(abjad.Block("layout"))
        score_block.items.append(abjad.Block("midi"))
        return abjad.LilyPondFile([score_block])

    def export_score_and_midi(
        self,
        score: abjad.Score,
        *,
        directory: pathlib.Path | None = None,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        directory = directory or pathlib.Path.cwd()
        ly_path = directory / f"{self.name}.ly"
        full_png = directory / f"{self.name}.png"
        cropped_path = directory / f"{self.name}.cropped.png"
        png_path = directory / f"{self.name}.png"
        midi_path = directory / f"{self.name}.mid"

        ly_file = self.prepare_ly_file(score)
        abjad.persist.as_ly(ly_file, str(ly_path))  # type: ignore[no-untyped-call]

        flags = "--png -dresolution=250 -dcrop"
        abjad.io.run_lilypond(str(ly_path), flags=flags)

        if cropped_path.exists():
            full_png.unlink(missing_ok=True)
            cropped_path.rename(png_path)

        midi_raw = directory / f"{self.name}.midi"
        if midi_raw.exists():
            midi_raw.rename(midi_path)

        midi_raw = directory / f"{self.name}.midi"
        if midi_raw.exists():
            midi_raw.rename(midi_path)

        return png_path, midi_path

    def export_score(
        self,
        score: abjad.Score | abjad.LilyPondFile,
        *,
        directory: pathlib.Path | None = None,
    ) -> pathlib.Path:
        directory = directory or pathlib.Path.cwd()
        original_path = directory / f"{self.name}_uncropped.png"
        path = directory / f"{self.name}.png"

        try:
            abjad.persist.as_png(
                score,
                str(original_path),
                resolution=250,
                flags="--png -dcrop",
            )
        except AttributeError:
            pass

        cropped_path = original_path.with_name(f"{original_path.stem}.cropped.png")
        cropped_path.rename(path)

        return path

    def export_midi(
        self,
        score: abjad.Score | abjad.LilyPondFile,
        *,
        directory: pathlib.Path | None = None,
    ) -> pathlib.Path:
        directory = directory or pathlib.Path.cwd()
        original_path = directory / f"{self.name}.midi"
        path = original_path.with_suffix(".mid")

        ly_file = Exporter.prepare_ly_file(score) if isinstance(score, abjad.Score) else score
        abjad.persist.as_midi(ly_file, str(original_path), remove_ly=False)

        original_path.rename(path)

        return path

    def export_audio(
        self,
        midi_path: pathlib.Path,
        audio_path: pathlib.Path,
    ) -> pathlib.Path:
        self.to_audio(self.soundfont_path, midi_path, audio_path, out_type=self.audio_format)
        if self.convert_to_mp3:
            mp3_path = audio_path.with_suffix(".mp3")
            sound = pydub.AudioSegment.from_wav(str(audio_path))
            sound.export(str(mp3_path), format="mp3")
            return mp3_path
        return audio_path

    def export(
        self,
        score: abjad.Score,
        directory: pathlib.Path,
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        if not isinstance(score, abjad.Score):
            raise TypeError(f"expected abjad.Score, got {type(score)}")

        image_path = pathlib.Path()
        midi_path = pathlib.Path()
        mp3_path = pathlib.Path()

        if not self.ignore_score or not self.ignore_midi:
            image_path, midi_path = self.export_score_and_midi(score, directory=directory)

        if not self.ignore_audio and midi_path != pathlib.Path():
            audio_path = directory / f"{self.name}.wav"
            mp3_path = self.export_audio(midi_path, audio_path)

        return image_path, midi_path, mp3_path

    def to_audio(
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
