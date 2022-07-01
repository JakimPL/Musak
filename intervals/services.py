import os
from typing import Any, Union

import yaml

from shared.dict import get_key


def default_settings(form: bool = False) -> dict[str, Any]:
    config_path = os.path.join('config', 'intervals.yml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        settings = config['default_settings']

    return settings


def get_settings(data: dict[str, Union[str, list]], intervals_definitions: dict[str, int]) -> dict[str, Any]:
    options = {key: True for key in data if get_key(data, key) == 'on'}
    settings = {
        'tempo': int(get_key(data, 'tempo')),
        'lowest_note': int(get_key(data, 'lowest_note')),
        'highest_note': int(get_key(data, 'highest_note')),
        'sequential': options.get('sequential', False)
    }

    intervals = {}
    for key, value in options.items():
        if 'interval_' in key:
            interval_name = key[9:]
            interval = intervals_definitions[interval_name]
            intervals[interval_name] = interval

    settings['intervals'] = intervals

    return settings


def get_intervals_definitions() -> dict[str, int]:
    config_path = os.path.join('config', 'intervals.yml')
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['intervals_definitions']
