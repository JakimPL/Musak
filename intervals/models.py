import json
import os

from chordinversions.exporter import to_abjad
from chordinversions.generator import get_random_interval
from chordinversions.interval import Interval
from chordinversions.inversion import ChordInversion

from intervals.services import get_intervals_definitions
from shared.directory import create_directory
from shared.exporter import Exporter


class IntervalModel:
    def __init__(self, settings: dict):
        intervals = settings['intervals']
        self._intervals: dict[str, int] = intervals if intervals else get_intervals_definitions()
        self._settings: dict = settings

    def get_random_chord_inversion(self) -> Interval:
        return get_random_interval(
            self._intervals,
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
        exporter = Exporter('interval', ignore_score=True)

        self.export_info(chord_inversion, os.path.join(path, 'interval.json'))
        exporter.export(score, path)

    def generate(self) -> str:
        uuid64, directory = create_directory()
        self.export_chord_inversion(directory)
        return uuid64

    @property
    def intervals(self) -> dict[str, int]:
        return self._intervals

    @property
    def inversions(self) -> dict[str, list[ChordInversion]]:
        return self._inversions
