from chordinversions.exporter import Exporter
from chordinversions.generator import get_random_chord_inversion, generate_all_inversions
from chordinversions.inversion import ChordInversion

from inversions.services import get_chords_definitions
from shared.directory import create_directory


class ChordInversionModel:
    def __init__(self, settings: dict):
        chords = settings['chords']
        self._chords = chords if chords else get_chords_definitions()
        self._inversions = generate_all_inversions(self._chords)
        self._settings = settings
        self._exporter = Exporter(
            sequential=settings['sequential'],
            tempo=settings['tempo']
        )

    def get_random_chord_inversion(self) -> ChordInversion:
        return get_random_chord_inversion(
            self._inversions,
            lowest_note=self._settings['lowest_note'],
            highest_note=self._settings['highest_note']
        )

    def export_chord_inversion(self, path: str, chord_inversion: ChordInversion = None):
        if chord_inversion is None:
            chord_inversion = self.get_random_chord_inversion()

        self._exporter.export(chord_inversion, path)

    def generate(self) -> str:
        uuid64, directory = create_directory()
        self.export_chord_inversion(directory)
        return uuid64

    @property
    def chords(self) -> dict[str, list[int]]:
        return self._chords
