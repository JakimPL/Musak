from typing import Any, Union

from config.defaults import TEMPO
from shared.dict import get_key


def default_settings(form: bool = False) -> dict[str, Any]:
    settings = {
        'sequential': False,
        'tempo': TEMPO
    }

    return settings


def get_settings(data: dict[str, Union[str, list]]) -> dict[str, Any]:
    options = {key: True for key in data if get_key(data, key) == 'on'}
    settings = {
        'tempo': int(get_key(data, 'tempo')),
        'lowest_note': int(get_key(data, 'lowest_note')),
        'highest_note': int(get_key(data, 'highest_note')),
        'sequential': options.get('sequential', False)
    }

    return settings
