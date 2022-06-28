import json
import os

from chordinversions.exporter import to_abjad
from chordinversions.generator import get_random_chord_inversion, generate_all_inversions
from chordinversions.inversion import ChordInversion

from inversions.services import get_chords_definitions
from shared.directory import create_directory
from shared.exporter import Exporter


class ChordInversionModel:
    def __init__(self, settings: dict):
        chords = settings['chords']
        self._chords: dict[str, list[int]] = chords if chords else get_chords_definitions()
        self._inversions: dict[str, list[ChordInversion]] = generate_all_inversions(self._chords)
        self._settings: dict = settings

    def get_max_inversion_index(self) -> int:
        return max([len(chord) for chord in self._inversions.values()])

    def get_random_chord_inversion(self) -> ChordInversion:
        return get_random_chord_inversion(
            self._inversions,
            lowest_note=self._settings['lowest_note'],
            highest_note=self._settings['highest_note']
        )

    @staticmethod
    def export_info(chord_inversion: ChordInversion, chord_info_path: str):
        with open(chord_info_path, 'w') as file:
            data = chord_inversion._asdict()
            data['base_note'] = chord_inversion.get_base_note_name()
            json.dump(data, file)

    def export_chord_inversion(self, path: str, chord_inversion: ChordInversion = None):
        if chord_inversion is None:
            chord_inversion = self.get_random_chord_inversion()

        score = to_abjad(chord_inversion.chord, self._settings['tempo'], self._settings['sequential'])
        exporter = Exporter('chord', ignore_score=True)

        self.export_info(chord_inversion, os.path.join(path, 'chord.json'))
        exporter.export(score, path)

    def generate(self) -> str:
        uuid64, directory = create_directory()
        self.export_chord_inversion(directory)
        return uuid64

    @property
    def chords(self) -> dict[str, list[int]]:
        return self._chords

    @property
    def inversions(self) -> dict[str, list[ChordInversion]]:
        return self._inversions
