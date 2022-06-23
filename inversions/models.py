from inversions.services import ChordInversionGeneratorService
from shared.directory import create_directory


class ChordInversionModel:
    def __init__(self):
        self.chord_inversion_generator_service = ChordInversionGeneratorService()

    def generate(self) -> str:
        uuid64, directory = create_directory()
        self.chord_inversion_generator_service.export_chord_inversion(directory)
        return uuid64
