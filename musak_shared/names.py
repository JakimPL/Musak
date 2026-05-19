from musak_shared.elements import FLAT_PITCH_CLASS_NAMES, KEYS, MIDDLE_C, SHARP_PITCH_CLASS_NAMES


def get_note_name(note: int, keys: dict[int, str] | None = None) -> str:
    if keys is None:
        keys = KEYS

    return keys[(note - MIDDLE_C) % 12]


def midi_to_vexflow_key(midi_note: int, *, prefer_flats: bool = False) -> str:
    names = FLAT_PITCH_CLASS_NAMES if prefer_flats else SHARP_PITCH_CLASS_NAMES
    pitch_name = names[midi_note % 12].lower()
    octave = (midi_note // 12) - 1
    return f"{pitch_name}/{octave}"
