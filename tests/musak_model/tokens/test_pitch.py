from musak_model.tokens.pitch import note_token_to_midi_pitch
from musak_model.tokens.schema import Hand, NoteToken, ScaleType


def test_note_token_to_midi_pitch_uses_key_scale_and_hand_register() -> None:
    token = NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=0)

    assert (
        note_token_to_midi_pitch(
            token,
            key_root=0,
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
            key_root=0,
            scale_type=ScaleType.MAJOR,
            hand=Hand.LEFT,
        )
        == 61
    )
