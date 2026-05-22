from musak_model.tokens.schema import Hand, HandToken, HoldToken, NoteToken, RestToken


def construct_note_token(
    *,
    degree: int,
    accidental: int,
    octave_offset: int,
    duration_id: int,
) -> NoteToken:
    return NoteToken.model_construct(
        kind="note",
        degree=degree,
        accidental=accidental,
        octave_offset=octave_offset,
        duration_id=duration_id,
    )


def construct_rest_token(*, duration_id: int) -> RestToken:
    return RestToken.model_construct(kind="rest", duration_id=duration_id)


def construct_hold_token(*, duration_id: int) -> HoldToken:
    return HoldToken.model_construct(kind="hold", duration_id=duration_id)


def construct_hand_token(*, hand: Hand) -> HandToken:
    return HandToken.model_construct(kind="hand", hand=hand)
