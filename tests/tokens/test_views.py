from fractions import Fraction

from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, NoteToken, RestToken
from musak_model.tokens.views import tokens_for_hand


def test_tokens_for_hand_selects_active_hand_tokens(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 8))
    right_note = NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id)
    left_note = NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=eighth_id)
    tokens = [
        HandToken(hand=Hand.RIGHT),
        right_note,
        BarToken(),
        HandToken(hand=Hand.LEFT),
        RestToken(duration_id=quarter_id),
        left_note,
        EndToken(),
    ]

    assert tokens_for_hand(tokens, hand=Hand.RIGHT) == [right_note, BarToken(), EndToken()]
    assert tokens_for_hand(tokens, hand=Hand.LEFT, include_structure=False) == [
        RestToken(duration_id=quarter_id),
        left_note,
    ]
