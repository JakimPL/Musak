from musak_model.tokens.factory import (
    construct_hand_token,
    construct_hold_token,
    construct_note_token,
    construct_rest_token,
)
from musak_model.tokens.schema import Hand, HandToken, HoldToken, NoteToken, RestToken


def test_construct_note_token_matches_validated_model() -> None:
    assert construct_note_token(degree=1, accidental=0, octave_offset=1, duration_id=2) == NoteToken(
        degree=1,
        accidental=0,
        octave_offset=1,
        duration_id=2,
    )


def test_construct_rest_token_matches_validated_model() -> None:
    assert construct_rest_token(duration_id=2) == RestToken(duration_id=2)


def test_construct_hold_token_matches_validated_model() -> None:
    assert construct_hold_token(duration_id=2) == HoldToken(duration_id=2)


def test_construct_hand_token_matches_validated_model() -> None:
    assert construct_hand_token(hand=Hand.RIGHT) == HandToken(hand=Hand.RIGHT)
