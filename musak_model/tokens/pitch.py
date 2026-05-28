from musak_model.tokens.schema import HAND_HOME_OCTAVES, SCALE_INTERVALS, Hand, NoteToken, ScaleType
from musak_shared.elements import MIDI_OCTAVE_OFFSET, PITCHES_PER_OCTAVE


def degree_pitch_class(degree: int, accidental: int, *, scale_type: ScaleType) -> int:
    return (SCALE_INTERVALS[scale_type][degree - 1] + accidental) % PITCHES_PER_OCTAVE


def note_token_to_midi_pitch(
    token: NoteToken,
    *,
    scale_root: int,
    scale_type: ScaleType,
    hand: Hand,
) -> int:
    interval_class = degree_pitch_class(token.degree, token.accidental, scale_type=scale_type)
    pitch_class = (scale_root + interval_class) % PITCHES_PER_OCTAVE
    octave = HAND_HOME_OCTAVES[hand] + token.octave_offset
    return (octave + MIDI_OCTAVE_OFFSET) * PITCHES_PER_OCTAVE + pitch_class


def note_diatonic_position(token: NoteToken, *, scale_size: int) -> int:
    if scale_size <= 0:
        raise ValueError("scale_size must be positive")

    return token.octave_offset * scale_size + (token.degree - 1)


def diatonic_position_to_degree_and_octave(position: int, *, scale_size: int) -> tuple[int, int]:
    if scale_size <= 0:
        raise ValueError("scale_size must be positive")

    return position % scale_size + 1, position // scale_size
