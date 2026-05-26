from musak_model.n_grams.figure.pitch import note_diatonic_position
from musak_model.tokens.schema import NoteToken


def test_note_diatonic_position_uses_configured_scale_size() -> None:
    token = NoteToken(degree=3, accidental=0, octave_offset=2, duration_id=0)

    assert note_diatonic_position(token, scale_size=7) == 16
    assert note_diatonic_position(token, scale_size=5) == 12
