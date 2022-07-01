import re
from typing import Any, Union, Optional

import abjad

from config.defaults import TEMPO, GROUPS, MEASURES
from modules.rhythm.exceptions import RhygenException
from modules.rhythm.generator import RhythmGenerator
from modules.rhythm.settings import Settings
from shared.dict import get_key
from shared.directory import create_directory
from shared.exporter import Exporter

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


def default_settings(form: bool = False) -> dict[str, Any]:
    settings = {
        'groups': GROUPS,
        'measures': MEASURES,
        'tempo': TEMPO,
        'time_signature_numerator': 4,
        'time_signature_denominator': 4,
        'half_note': 'on',
        'half_rest': 'on',
        'quarter_note': 'on'
    }

    return settings if form else get_settings(settings)


def parse_custom_phrase(raw_phrase: str) -> list:
    elements = raw_phrase.split(',')
    notes = []

    for element in elements:
        if '(' in element or ')' in element:
            if re.match(r"(-)?\(\d+:\d+\)", element):
                raw_pair = element.replace('(', '').replace(')', '').split(':')
                note = int(raw_pair[0]), int(raw_pair[1])
            else:
                raise ValueError(f'invalid note element {element}')
        else:
            note = int(element)

        notes.append(note)

    return notes


def parse_custom_phrases(phrases_string: str) -> list:
    if not phrases_string:
        return []

    string = phrases_string.strip().replace(' ', '')

    count = 0
    elements = []
    raw_phrase = ''
    for char in string:
        if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '[', ']', '(', ')', '-', ':']:
            raise ValueError(f'unexpected symbol {char}')
        bracket = True
        if char == '[':
            count += 1
        elif char == ']':
            count -= 1
            if raw_phrase:
                elements.append(raw_phrase)

            raw_phrase = ''
        else:
            bracket = False
            if count != 0:
                raw_phrase += char

        if count < 0:
            raise ValueError('unexpected closing bracket')
        elif count > 1:
            raise ValueError('unexpected nested expression')
        elif count == 0 and char != ',' and not bracket:
            raise ValueError(f'unexpected symbol {char} outside the expression')

    if count != 0:
        raise ValueError('unbalanced square brackets')

    return [parse_custom_phrase(raw_phrase) for raw_phrase in elements]


def get_settings(data: dict[str, Union[str, list]]) -> dict[str, Any]:
    settings = {
        'notes': [],
        'phrases': [],
        'tempo': int(get_key(data, 'tempo')),
        'groups': int(get_key(data, 'groups')),
        'measures': int(get_key(data, 'measures')),
        'time_signature': (
            int(get_key(data, 'time_signature_numerator')),
            int(get_key(data, 'time_signature_denominator'))
        )}

    options = {key: get_key(data, key) for key in data if get_key(data, key) == 'on'}
    for key, value in options.items():
        if '_phrase' in key:
            settings['phrases'].append(settings_map[key])
        else:
            settings['notes'].append(settings_map[key])

    settings['phrases'] += parse_custom_phrases(get_key(data, 'custom_phrases'))

    return settings


class RhygenService:
    def __init__(self, settings: dict):
        self.rhythm_generator = RhythmGenerator()
        self.time_signature_error = False
        self.exception = None
        self.score: Optional[abjad.Score] = None
        self.uuid: str = ''
        self.image: str = ''
        self.midi: str = ''
        self.audio: str = ''

        self.create_settings(settings)
        self.create_score()

    def create_settings(self, settings: dict):
        try:
            self.settings = settings
        except ValueError:
            settings['time_signature'] = (4, 4)
            self.settings = settings
            self.time_signature_error = True

    def create_score(self):
        try:
            self.score, self.uuid, self.image, self.midi, self.audio = self.generate()
        except RhygenException as exception:
            self.exception = exception

    @property
    def settings(self) -> Settings:
        return self.rhythm_generator.settings

    @settings.setter
    def settings(self, dictionary: dict):
        tempo = dictionary['tempo']
        groups = dictionary['groups']
        measures = dictionary['measures']
        notes = dictionary['notes']
        phrases = dictionary['phrases']
        time_signature = dictionary['time_signature']

        self.rhythm_generator.settings.groups = groups
        self.rhythm_generator.settings.measures = measures

        self.rhythm_generator.settings.tempo = tempo
        self.rhythm_generator.settings.time_signature = time_signature

        self.rhythm_generator.settings.default_group_settings.notes = notes
        self.rhythm_generator.settings.default_group_settings.phrases = phrases

    def generate(self) -> tuple[abjad.Score, str, str, str, str]:
        exporter = Exporter('rhythm')

        uuid64, directory = create_directory()
        rhythm = self.rhythm_generator()

        image, midi, audio = exporter.export(rhythm, directory)
        return rhythm, uuid64, image, midi, audio
