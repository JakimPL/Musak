from fractions import Fraction

import pytest

from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration_vocabulary import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, JoinWithPreviousToken, NoteToken
from musak_model.tokens.vocabulary import TokenVocabulary


def _vocabulary() -> TokenVocabulary:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1))
    return TokenVocabulary(duration_vocabulary)


def test_structural_tokens_roundtrip() -> None:
    vocabulary = _vocabulary()
    tokens = [HandToken(hand=Hand.RIGHT), HandToken(hand=Hand.LEFT), JoinWithPreviousToken()]

    token_ids = vocabulary.encode(tokens)

    assert vocabulary.decode(token_ids) == tokens


def test_note_token_roundtrip_still_works() -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1))
    vocabulary = TokenVocabulary(duration_vocabulary)
    token = NoteToken(
        degree=1,
        accidental=0,
        octave_offset=0,
        duration_id=duration_vocabulary.fraction_to_id(Fraction(1, 4)),
    )

    assert vocabulary.id_to_token(vocabulary.token_to_id(token)) == token


def test_rejects_token_id_outside_extended_vocabulary() -> None:
    vocabulary = _vocabulary()

    with pytest.raises(ValueError, match="token_id"):
        vocabulary.id_to_token(vocabulary.vocab_size)
