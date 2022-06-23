from chordinversions.exporter import Exporter
from chordinversions.generator import get_random_chord_inversion, generate_all_inversions
from chordinversions.inversion import ChordInversion


class ChordInversionGeneratorService:
    def __init__(self):
        self._inversions = generate_all_inversions()
        self._exporter = Exporter()

    def get_random_chord_inversion(self) -> ChordInversion:
        return get_random_chord_inversion(self._inversions)

    def export_chord_inversion(self, path: str, chord_inversion: ChordInversion = None):
        if chord_inversion is None:
            chord_inversion = self.get_random_chord_inversion()

        self._exporter.export(chord_inversion, path)
