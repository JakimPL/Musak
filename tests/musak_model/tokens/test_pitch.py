import pytest

from musak_model.tokens.pitch import (
    diatonic_position_to_degree_and_octave,
    note_diatonic_position,
    note_token_to_midi_pitch,
)
from musak_model.tokens.schema import Hand, NoteToken, ScaleType


def test_note_token_to_midi_pitch_uses_key_scale_and_hand_register() -> None:
    token = NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=0)

    assert (
        note_token_to_midi_pitch(
            token,
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            hand=Hand.RIGHT,
        )
        == 79
    )


def test_note_token_to_midi_pitch_applies_accidental_and_octave_offset() -> None:
    token = NoteToken(degree=1, accidental=1, octave_offset=1, duration_id=0)

    assert (
        note_token_to_midi_pitch(
            token,
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            hand=Hand.LEFT,
        )
        == 61
    )


def test_note_diatonic_position_uses_configured_scale_size() -> None:
    token = NoteToken(degree=3, accidental=0, octave_offset=2, duration_id=0)

    assert note_diatonic_position(token, scale_size=7) == 16
    assert note_diatonic_position(token, scale_size=5) == 12


def test_diatonic_position_to_degree_and_octave_inverts_note_diatonic_position() -> None:
    for octave_offset in range(-2, 3):
        for degree in range(1, 8):
            token = NoteToken(degree=degree, accidental=0, octave_offset=octave_offset, duration_id=0)
            position = note_diatonic_position(token, scale_size=7)
            recovered_degree, recovered_octave = diatonic_position_to_degree_and_octave(position, scale_size=7)
            assert (recovered_degree, recovered_octave) == (degree, octave_offset)


def test_diatonic_position_helpers_reject_non_positive_scale_size() -> None:
    token = NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0)

    with pytest.raises(ValueError, match="scale_size"):
        note_diatonic_position(token, scale_size=0)

    with pytest.raises(ValueError, match="scale_size"):
        diatonic_position_to_degree_and_octave(0, scale_size=0)
