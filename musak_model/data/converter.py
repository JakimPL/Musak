from musak_model.common.elements import MIDI_OCTAVE_OFFSET, PITCHES_PER_OCTAVE
from musak_model.data.schema import PitchDegree
from musak_model.tokens.schema import (
    HAND_HOME_OCTAVES,
    SCALE_INTERVALS,
    Hand,
    ScaleType,
)


def pitch_to_degree(
    midi_pitch: int,
    *,
    key_root: int,
    scale_type: ScaleType,
    hand: Hand,
) -> PitchDegree:
    pitch_class = midi_pitch % PITCHES_PER_OCTAVE
    degree, accidental = _pitch_class_to_degree(pitch_class, key_root=key_root, scale_type=scale_type)
    octave_offset = _compute_octave_offset(midi_pitch, hand=hand)
    return PitchDegree(degree=degree, accidental=accidental, octave_offset=octave_offset)


def _pitch_class_to_degree(
    pitch_class: int,
    *,
    key_root: int,
    scale_type: ScaleType,
) -> tuple[int, int]:
    intervals = SCALE_INTERVALS[scale_type]
    relative = (pitch_class - key_root) % PITCHES_PER_OCTAVE

    if relative in intervals:
        return intervals.index(relative) + 1, 0

    for index, interval in enumerate(intervals):
        if (interval + 1) % PITCHES_PER_OCTAVE == relative:
            return index + 1, 1

    for index, interval in enumerate(intervals):
        if (interval - 1) % PITCHES_PER_OCTAVE == relative:
            return index + 1, -1

    raise ValueError(f"cannot map pitch class {pitch_class} to a degree in key {key_root} scale {scale_type.value}")


def _compute_octave_offset(
    midi_pitch: int,
    *,
    hand: Hand,
) -> int:
    octave = midi_pitch // PITCHES_PER_OCTAVE - MIDI_OCTAVE_OFFSET
    return octave - HAND_HOME_OCTAVES[hand]
