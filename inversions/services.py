import os
from typing import Any, Union

import yaml

from shared.dict import get_key


def default_settings(form: bool = False) -> dict[str, Any]:
    config_path = os.path.join('config', 'inversions.yml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        settings = config['default_settings']

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


def get_chords_definitions() -> dict[str, list[int]]:
    config_path = os.path.join('config', 'inversions.yml')
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['chords_definitions']
