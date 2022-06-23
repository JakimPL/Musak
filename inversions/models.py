import os

import yaml
from chordinversions.exporter import Exporter
from chordinversions.generator import get_random_chord_inversion, generate_all_inversions
from chordinversions.inversion import ChordInversion

from config.defaults import TEMPO
from shared.directory import create_directory


class ChordInversionModel:
    def __init__(
            self,
            sequential: bool = True,
            tempo: int = TEMPO
    ):
        self._chords = self.get_chords_definitions()
        self._inversions = generate_all_inversions(self._chords)
        self._exporter = Exporter(
            sequential=sequential,
            tempo=tempo
        )

    @staticmethod
    def get_chords_definitions() -> dict:
        config_path = os.path.join('config', 'inversions.yml')
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
            return data['chord_definitions']

    def get_random_chord_inversion(self) -> ChordInversion:
        return get_random_chord_inversion(self._inversions)

    def export_chord_inversion(self, path: str, chord_inversion: ChordInversion = None):
        if chord_inversion is None:
            chord_inversion = self.get_random_chord_inversion()

        self._exporter.export(chord_inversion, path)

    def generate(self) -> str:
        uuid64, directory = create_directory()
        self.export_chord_inversion(directory)
        return uuid64
