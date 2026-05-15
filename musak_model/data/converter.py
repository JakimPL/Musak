from pydantic import ValidationError

from musak_model.common.elements import MIDI_OCTAVE_OFFSET, PITCHES_PER_OCTAVE
from musak_model.data.schema import PitchDegree
from musak_model.tokens.schema import (
    HAND_HOME_OCTAVES,
    SCALE_INTERVALS,
    Hand,
    ScaleType,
)


class PitchDegreeRegisterError(ValueError):
    def __init__(self, midi_pitch: int, *, hand: Hand, octave_offset: int) -> None:
        super().__init__(
            f"pitch {midi_pitch} maps to octave offset {octave_offset}, outside supported {hand.value} hand register"
        )
        self.midi_pitch = midi_pitch
        self.hand = hand
        self.octave_offset = octave_offset


def pitch_to_degree(
    midi_pitch: int,
    *,
    key_root: int,
    key_fifths: int,
    scale_type: ScaleType,
    hand: Hand,
) -> PitchDegree:
    pitch_class = midi_pitch % PITCHES_PER_OCTAVE
    degree, accidental = _pitch_class_to_degree(
        pitch_class,
        key_root=key_root,
        key_fifths=key_fifths,
        scale_type=scale_type,
    )
    octave_offset = _compute_octave_offset(midi_pitch, hand=hand)
    try:
        return PitchDegree(degree=degree, accidental=accidental, octave_offset=octave_offset)
    except ValidationError as exception:
        if _is_octave_offset_validation_error(exception):
            raise PitchDegreeRegisterError(midi_pitch, hand=hand, octave_offset=octave_offset) from exception
        raise


def _pitch_class_to_degree(
    pitch_class: int,
    *,
    key_root: int,
    key_fifths: int,
    scale_type: ScaleType,
) -> tuple[int, int]:
    intervals = SCALE_INTERVALS[scale_type]
    relative = (pitch_class - key_root) % PITCHES_PER_OCTAVE

    if relative in intervals:
        return intervals.index(relative) + 1, 0

    # Policy: prefer flat for key_fifths < 0, sharp otherwise
    checks = [(-1, -1), (1, 1)] if key_fifths < 0 else [(1, 1), (-1, -1)]
    for step, accidental in checks:
        for index, interval in enumerate(intervals):
            if (interval + step) % PITCHES_PER_OCTAVE == relative:
                return index + 1, accidental

    raise ValueError(
        f"cannot map pitch class {pitch_class} to a degree in key {key_root} "
        f"(fifths={key_fifths}) scale {scale_type.value}"
    )


def _compute_octave_offset(
    midi_pitch: int,
    *,
    hand: Hand,
) -> int:
    octave = midi_pitch // PITCHES_PER_OCTAVE - MIDI_OCTAVE_OFFSET
    return octave - HAND_HOME_OCTAVES[hand]


def _is_octave_offset_validation_error(exception: ValidationError) -> bool:
    return any(error["loc"] == ("octave_offset",) for error in exception.errors())
