from fractions import Fraction

import pytest

from musak_model.tokens.factorized import (
    ABSENT_ATTRIBUTE_ID,
    TokenAttributes,
    TokenKindId,
    attributes_to_token,
    attributes_to_token_id,
    flat_vocabulary_attributes,
    predicted_attributes_to_token,
    predicted_attributes_to_token_id,
    token_id_to_attributes,
    token_to_attributes,
)
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    StartToken,
)
from musak_model.tokens.vocabulary import TokenVocabulary


def test_factorized_attributes_round_trip_every_flat_vocabulary_id(token_vocabulary: TokenVocabulary) -> None:
    for token_id in range(token_vocabulary.vocabulary_size):
        attributes = token_id_to_attributes(token_id, vocabulary=token_vocabulary)

        assert attributes_to_token_id(attributes, vocabulary=token_vocabulary) == token_id


def test_flat_vocabulary_attribute_table_matches_token_vocabulary_order(
    token_vocabulary: TokenVocabulary,
) -> None:
    table = flat_vocabulary_attributes(duration_vocabulary_size=token_vocabulary.duration_vocabulary.vocabulary_size())

    assert len(table) == token_vocabulary.vocabulary_size
    assert list(table) == [
        token_id_to_attributes(token_id, vocabulary=token_vocabulary)
        for token_id in range(token_vocabulary.vocabulary_size)
    ]


def test_factorized_attributes_capture_note_fields(token_vocabulary: TokenVocabulary) -> None:
    token = NoteToken(degree=3, accidental=-1, octave_offset=2, duration_id=4)

    attributes = token_to_attributes(token)

    assert attributes == TokenAttributes(
        kind_id=TokenKindId.NOTE,
        degree_id=2,
        accidental_id=0,
        octave_offset_id=4,
        duration_id=4,
        hand_id=ABSENT_ATTRIBUTE_ID,
    )
    assert attributes_to_token(attributes) == token
    assert predicted_attributes_to_token_id(attributes, vocabulary=token_vocabulary) == token_vocabulary.token_to_id(
        token
    )


def test_factorized_attributes_capture_structural_tokens(token_vocabulary: TokenVocabulary) -> None:
    quarter_id = token_vocabulary.duration_vocabulary.fraction_to_id(Fraction(1, 4))
    tokens = [
        RestToken(duration_id=quarter_id),
        HoldToken(duration_id=quarter_id),
        BarToken(),
        EndToken(),
        HandToken(hand=Hand.LEFT),
        JoinWithPreviousToken(),
        StartToken(),
    ]

    assert [attributes_to_token(token_to_attributes(token)) for token in tokens] == tokens


def test_strict_reconstruction_rejects_inactive_attributes() -> None:
    attributes = TokenAttributes(
        kind_id=TokenKindId.REST,
        duration_id=0,
        degree_id=0,
    )

    with pytest.raises(ValueError, match="degree_id must be absent"):
        attributes_to_token(attributes)


def test_prediction_reconstruction_ignores_inactive_heads() -> None:
    attributes = TokenAttributes(
        kind_id=TokenKindId.REST,
        duration_id=0,
        degree_id=0,
        accidental_id=1,
        octave_offset_id=2,
    )

    assert predicted_attributes_to_token(attributes) == RestToken(duration_id=0)


def test_reconstruction_rejects_missing_active_attribute() -> None:
    attributes = TokenAttributes(kind_id=TokenKindId.NOTE, duration_id=0)

    with pytest.raises(ValueError, match="degree_id"):
        predicted_attributes_to_token(attributes)


def test_reconstruction_rejects_unknown_kind() -> None:
    attributes = TokenAttributes(kind_id=999)

    with pytest.raises(ValueError, match="unknown token kind"):
        predicted_attributes_to_token(attributes)
