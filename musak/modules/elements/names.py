from typing import Optional

from musak_shared.elements import KEYS


def get_note_name(note: int, keys: Optional[dict[int, str]] = None) -> str:
    if keys is None:
        keys = KEYS

    return keys[(note - 60) % 12]


def midi_to_vexflow_key(midi_note: int) -> str:
    pitch_name = KEYS[midi_note % 12].lower()
    octave = (midi_note // 12) - 1
    return f"{pitch_name}/{octave}"
