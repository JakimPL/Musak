from rhygen.generator import RhythmGenerator
from rhygen.modules.exceptions import RhygenException
from rhygen.modules.misc import save_score
from rhygen.modules.settings import Settings

from shared.directory import create_directory

settings_map = {
    'whole_note': 1,
    'half_note': 2,
    'quarter_note': 4,
    'eighth_note': 8,
    'sixteenth_note': 16,
    'thirty_second_note': 32,

    'whole_rest': -1,
    'half_rest': -2,
    'quarter_rest': -4,
    'eighth_rest': -8,
    'sixteenth_rest': -16,
    'thirty_second_rest': -32,

    'dotted_half_note': (3, 4),
    'dotted_quarter_note': (3, 8),
    'dotted_eighth_note': (3, 16),
    'dotted_sixteenth_note': (3, 32),

    'two_quarter_notes_phrase': [4, 4],
    'two_eighth_notes_phrase': [8, 8],
    'four_eighth_notes_phrase': [8, 8, 8, 8],
    'two_sixteenth_notes_phrase': [16, 16],
    'four_sixteenth_notes_phrase': [16, 16, 16, 16],
    'eight_sixteenth_notes_phrase': [16, 16, 16, 16, 16, 16, 16, 16],

    'left_quarter_phrase': [4, -4],
    'right_quarter_phrase': [-4, 4],

    'left_eighth_phrase': [8, -8],
    'right_eighth_phrase': [-8, 8],

    'left_sixteenth_phrase': [16, -16],
    'right_sixteenth_phrase': [-16, 16]
}


class RhygenService:
    def __init__(self, settings: dict):
        self.rhythm_generator = RhythmGenerator()

        self.time_signature_error = False
        try:
            self.settings = settings
        except ValueError:
            settings['time_signature'] = (4, 4)
            self.settings = settings
            self.time_signature_error = True

        self.exception = None
        self.score = None
        try:
            self.image = self.generate_image()
            self.score = self.rhythm_generator.cache
        except RhygenException as ex:
            self.image = ''
            self.exception = ex

    @property
    def settings(self) -> Settings:
        return self.rhythm_generator.settings

    @settings.setter
    def settings(self, dictionary: dict):
        groups = dictionary['groups']
        measures = dictionary['measures']
        notes = dictionary['notes']
        phrases = dictionary['phrases']
        time_signature = dictionary['time_signature']

        self.rhythm_generator.settings.groups = groups
        self.rhythm_generator.settings.measures = measures
        self.rhythm_generator.settings.time_signature = time_signature

        self.rhythm_generator.settings.default_group_settings.notes = notes
        self.rhythm_generator.settings.default_group_settings.phrases = phrases

    def generate_image(self):
        directory = create_directory()[1]
        return save_score(self.rhythm_generator(), directory, remove_ly=True, resolution=250, flags='--png -dcrop')
