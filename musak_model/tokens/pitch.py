from musak_model.tokens.schema import HAND_HOME_OCTAVES, SCALE_INTERVALS, Hand, NoteToken, ScaleType
from musak_shared.elements import MIDI_OCTAVE_OFFSET, PITCHES_PER_OCTAVE


def note_token_to_midi_pitch(
    token: NoteToken,
    *,
    scale_root: int,
    scale_type: ScaleType,
    hand: Hand,
) -> int:
    interval = SCALE_INTERVALS[scale_type][token.degree - 1]
    pitch_class = (scale_root + interval + token.accidental) % PITCHES_PER_OCTAVE
    octave = HAND_HOME_OCTAVES[hand] + token.octave_offset
    return (octave + MIDI_OCTAVE_OFFSET) * PITCHES_PER_OCTAVE + pitch_class


def note_token_to_static_hand_position(token: NoteToken) -> int:
    return token.octave_offset * 7 + token.degree
