from fractions import Fraction

import pytest

from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, JoinWithPreviousToken, NoteToken
from musak_model.tokens.vocabulary import TokenVocabulary


def test_structural_tokens_roundtrip(token_vocabulary: TokenVocabulary) -> None:
    tokens = [HandToken(hand=Hand.RIGHT), HandToken(hand=Hand.LEFT), JoinWithPreviousToken()]

    token_ids = token_vocabulary.encode(tokens)

    assert token_vocabulary.decode(token_ids) == tokens


def test_note_token_roundtrip_still_works(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    token = NoteToken(
        degree=1,
        accidental=0,
        octave_offset=0,
        duration_id=duration_vocabulary.fraction_to_id(Fraction(1, 4)),
    )

    assert token_vocabulary.id_to_token(token_vocabulary.token_to_id(token)) == token


def test_rejects_token_id_outside_extended_vocabulary(token_vocabulary: TokenVocabulary) -> None:
    with pytest.raises(ValueError, match="token_id"):
        token_vocabulary.id_to_token(token_vocabulary.vocabulary_size)
