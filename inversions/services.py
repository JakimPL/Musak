from typing import Any

from config.defaults import TEMPO


def default_settings(form: bool = False) -> dict[str, Any]:
    settings = {
        'sequential': False,
        'tempo': TEMPO
    }

    return settings
