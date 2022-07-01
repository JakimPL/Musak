from modules.chords.constants import KEYS


def get_note_name(note: int, keys: dict[int, str] = None) -> str:
    if keys is None:
        keys = KEYS

    return keys[(note - 60) % 12]
