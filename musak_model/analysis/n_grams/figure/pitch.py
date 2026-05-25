from musak_model.tokens.schema import NoteToken


def note_diatonic_position(
    token: NoteToken,
    *,
    scale_size: int,
) -> int:
    if scale_size <= 0:
        raise ValueError("scale_size must be positive")

    return token.octave_offset * scale_size + (token.degree - 1)
