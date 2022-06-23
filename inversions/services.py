import os
from typing import Any, Union

import yaml

from config.defaults import TEMPO, LOWEST_NOTE, HIGHEST_NOTE
from shared.dict import get_key


def default_settings(form: bool = False) -> dict[str, Any]:
    settings = {
        'sequential': False,
        'tempo': TEMPO,
        'lowest_note': LOWEST_NOTE,
        'highest_note': HIGHEST_NOTE
    }

    return settings


def get_settings(data: dict[str, Union[str, list]], chords_definitions: dict[str, list[int]]) -> dict[str, Any]:
    options = {key: True for key in data if get_key(data, key) == 'on'}
    settings = {
        'tempo': int(get_key(data, 'tempo')),
        'lowest_note': int(get_key(data, 'lowest_note')),
        'highest_note': int(get_key(data, 'highest_note')),
        'sequential': options.get('sequential', False)
    }

    chords = {}
    for key, value in options.items():
        if 'chord_' in key:
            chord_name = key[6:]
            chord = chords_definitions[chord_name]
            chords[chord_name] = chord

    settings['chords'] = chords

    return settings


def get_chords_definitions() -> dict:
    config_path = os.path.join('config', 'inversions.yml')
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['chord_definitions']
