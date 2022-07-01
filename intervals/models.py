import json
import os

from intervals.services import get_intervals_definitions
from modules.chords.exporter import to_abjad
from modules.chords.generator import get_random_interval
from modules.chords.interval import Interval
from shared.directory import create_directory
from shared.exporter import Exporter


class IntervalModel:
    def __init__(self, settings: dict):
        intervals = settings['intervals']
        self._intervals: dict[str, int] = intervals if intervals else get_intervals_definitions()
        self._settings: dict = settings

    def get_random_interval(self) -> Interval:
        return get_random_interval(
            self._intervals,
            lowest_note=self._settings['lowest_note'],
            highest_note=self._settings['highest_note']
        )

    @staticmethod
    def export_info(interval: Interval, interval_info_path: str):
        with open(interval_info_path, 'w') as file:
            data = interval._asdict()
            data['base_note'] = interval.get_base_note_name()
            data['name'] = interval.name
            json.dump(data, file)

    def export_interval(self, path: str, interval: Interval = None):
        if interval is None:
            interval = self.get_random_interval()

        score = to_abjad(interval.chord, self._settings['tempo'], self._settings['sequential'])
        exporter = Exporter('interval')

        self.export_info(interval, os.path.join(path, 'interval.json'))
        exporter.export(score, path)

    def generate(self) -> str:
        uuid64, directory = create_directory()
        self.export_interval(directory)
        return uuid64

    @property
    def intervals(self) -> dict[str, int]:
        return self._intervals
