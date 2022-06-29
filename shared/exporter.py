import os
import pathlib
import subprocess
from typing import Union

import abjad
import pydub

SOUNDFONT = 'st_concert.sf2'
AUDIO_FORMAT = 'wav'
CONVERT_TO_MP3 = True
IGNORE_MIDI = False
IGNORE_AUDIO = False
IGNORE_SCORE = True
GAIN = 5.0


class Exporter:
    def __init__(
            self,
            name: str,
            sf2: str = SOUNDFONT,
            ignore_midi: bool = IGNORE_MIDI,
            ignore_audio: bool = IGNORE_AUDIO,
            ignore_score: bool = IGNORE_SCORE,
            audio_format: str = AUDIO_FORMAT,
            convert_to_mp3: bool = CONVERT_TO_MP3,
            gain: float = GAIN
    ):
        self.name: str = name

        self.soundfont_path: str = os.path.join(os.getcwd(), 'soundfont', sf2)
        self.audio_format: str = audio_format
        self.convert_to_mp3: bool = convert_to_mp3
        self.gain: float = gain

        self.ignore_midi: bool = ignore_midi
        self.ignore_audio: bool = ignore_audio
        self.ignore_score: bool = ignore_score

    @staticmethod
    def prepare_ly_file(score: abjad.Score) -> abjad.LilyPondFile:
        score_block = abjad.Block('score', items=[score])
        midi_block = abjad.Block('midi')
        score_block.items.append(midi_block)
        return abjad.LilyPondFile([score_block])

    def export_score(self, score: Union[abjad.Score, abjad.LilyPondFile], directory: str = None, **kwargs) -> str:
        if directory is None:
            directory = os.getcwd()

        original_path = os.path.join(directory, '{name}_uncropped.png'.format(name=self.name))
        path = os.path.join(directory, '{name}.png'.format(name=self.name))

        try:
            abjad.persist.as_png(score, original_path, remove_ly=True, resolution=250, flags='--png -dcrop', **kwargs)
        except AttributeError:
            pass

        cropped_path = original_path[:-4] + '.cropped.png'
        os.rename(cropped_path, path)

        return path

    def export_midi(self, score: Union[abjad.Score, abjad.LilyPondFile], directory: str = None, **kwargs) -> str:
        if directory is None:
            directory = os.getcwd()

        original_path = os.path.join(directory, '{name}.midi'.format(name=self.name))
        path = str(pathlib.Path(original_path).with_suffix('.mid'))

        ly_file = Exporter.prepare_ly_file(score) if isinstance(score, abjad.Score) else score
        abjad.persist.as_midi(ly_file, original_path, remove_ly=False, **kwargs)

        os.rename(original_path, path)

        return path

    def export_audio(self, midi_path: str, audio_path: str) -> str:
        self.to_audio(self.soundfont_path, midi_path, audio_path, out_type=self.audio_format)
        if self.convert_to_mp3:
            mp3_path = str(pathlib.Path(audio_path).with_suffix('.mp3'))
            sound = pydub.AudioSegment.from_wav(audio_path)
            sound.export(mp3_path, format='mp3')
            return mp3_path
        else:
            return audio_path

    def export(self, score: abjad.Score, directory: str) -> tuple[str, str, str]:
        if not isinstance(score, abjad.Score):
            raise TypeError('expected abjad.Score, got {type}'.format(type=type(score)))

        image_path = ''
        midi_path = ''
        mp3_path = ''

        if not self.ignore_score:
            image_path = self.export_score(score, directory)

        if not self.ignore_midi:
            midi_path = self.export_midi(score, directory)

            if not self.ignore_audio:
                audio_path = os.path.join(directory, '{name}.wav'.format(name=self.name))
                mp3_path = self.export_audio(midi_path, audio_path)

        return image_path, midi_path, mp3_path

    def to_audio(self, sf2: str, midi_file: str, out_file: str, out_type: str = 'wav'):
        subprocess.call(['fluidsynth', '-g', str(self.gain), '-T', out_type, '-F', out_file, '-ni', sf2, midi_file])

