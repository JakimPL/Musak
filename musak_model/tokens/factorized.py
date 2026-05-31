from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from musak_model.tokens.schema import (
    MAX_ACCIDENTAL,
    MAX_DEGREE,
    MAX_OCTAVE_OFFSET,
    MIN_ACCIDENTAL,
    MIN_DEGREE,
    MIN_DURATION_ID,
    MIN_OCTAVE_OFFSET,
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    StartToken,
    Token,
)
from musak_model.tokens.vocabulary import TokenVocabulary

ABSENT_ATTRIBUTE_ID: Final[int] = -1

DEGREE_ATTRIBUTE_COUNT: Final[int] = MAX_DEGREE - MIN_DEGREE + 1
ACCIDENTAL_ATTRIBUTE_COUNT: Final[int] = MAX_ACCIDENTAL - MIN_ACCIDENTAL + 1
OCTAVE_OFFSET_ATTRIBUTE_COUNT: Final[int] = MAX_OCTAVE_OFFSET - MIN_OCTAVE_OFFSET + 1
HAND_ATTRIBUTE_COUNT: Final[int] = 2

_RIGHT_HAND_ID: Final[int] = 0
_LEFT_HAND_ID: Final[int] = 1
_HAND_ID_BY_HAND: Final[dict[Hand, int]] = {
    Hand.RIGHT: _RIGHT_HAND_ID,
    Hand.LEFT: _LEFT_HAND_ID,
}
_HAND_BY_ID: Final[dict[int, Hand]] = {
    _RIGHT_HAND_ID: Hand.RIGHT,
    _LEFT_HAND_ID: Hand.LEFT,
}


class TokenKindId(IntEnum):
    NOTE = 0
    REST = 1
    HOLD = 2
    BAR = 3
    END = 4
    HAND = 5
    JOIN_WITH_PREVIOUS = 6
    START = 7


TOKEN_KIND_COUNT: Final[int] = len(TokenKindId)


@dataclass(frozen=True)
class TokenAttributes:
    kind_id: int
    degree_id: int = ABSENT_ATTRIBUTE_ID
    accidental_id: int = ABSENT_ATTRIBUTE_ID
    octave_offset_id: int = ABSENT_ATTRIBUTE_ID
    duration_id: int = ABSENT_ATTRIBUTE_ID
    hand_id: int = ABSENT_ATTRIBUTE_ID


def token_to_attributes(token: Token) -> TokenAttributes:
    match token:
        case NoteToken():
            return TokenAttributes(
                kind_id=TokenKindId.NOTE,
                degree_id=degree_to_attribute_id(token.degree),
                accidental_id=accidental_to_attribute_id(token.accidental),
                octave_offset_id=octave_offset_to_attribute_id(token.octave_offset),
                duration_id=token.duration_id,
            )
        case RestToken():
            return TokenAttributes(kind_id=TokenKindId.REST, duration_id=token.duration_id)
        case HoldToken():
            return TokenAttributes(kind_id=TokenKindId.HOLD, duration_id=token.duration_id)
        case BarToken():
            return TokenAttributes(kind_id=TokenKindId.BAR)
        case EndToken():
            return TokenAttributes(kind_id=TokenKindId.END)
        case HandToken():
            return TokenAttributes(kind_id=TokenKindId.HAND, hand_id=hand_to_attribute_id(token.hand))
        case JoinWithPreviousToken():
            return TokenAttributes(kind_id=TokenKindId.JOIN_WITH_PREVIOUS)
        case StartToken():
            return TokenAttributes(kind_id=TokenKindId.START)


def token_id_to_attributes(token_id: int, *, vocabulary: TokenVocabulary) -> TokenAttributes:
    return token_to_attributes(vocabulary.id_to_token(token_id))


def token_ids_to_attributes(token_ids: list[int], *, vocabulary: TokenVocabulary) -> list[TokenAttributes]:
    return [token_id_to_attributes(token_id, vocabulary=vocabulary) for token_id in token_ids]


def attributes_to_token(attributes: TokenAttributes) -> Token:
    token = predicted_attributes_to_token(attributes)
    _validate_inactive_attributes_absent(attributes)
    return token


def predicted_attributes_to_token(attributes: TokenAttributes) -> Token:
    kind_id = _token_kind_id(attributes.kind_id)
    match kind_id:
        case TokenKindId.NOTE:
            return NoteToken(
                degree=attribute_id_to_degree(attributes.degree_id),
                accidental=attribute_id_to_accidental(attributes.accidental_id),
                octave_offset=attribute_id_to_octave_offset(attributes.octave_offset_id),
                duration_id=_duration_id(attributes.duration_id),
            )
        case TokenKindId.REST:
            return RestToken(duration_id=_duration_id(attributes.duration_id))
        case TokenKindId.HOLD:
            return HoldToken(duration_id=_duration_id(attributes.duration_id))
        case TokenKindId.BAR:
            return BarToken()
        case TokenKindId.END:
            return EndToken()
        case TokenKindId.HAND:
            return HandToken(hand=attribute_id_to_hand(attributes.hand_id))
        case TokenKindId.JOIN_WITH_PREVIOUS:
            return JoinWithPreviousToken()
        case TokenKindId.START:
            return StartToken()


def attributes_to_token_id(attributes: TokenAttributes, *, vocabulary: TokenVocabulary) -> int:
    return vocabulary.token_to_id(attributes_to_token(attributes))


def predicted_attributes_to_token_id(attributes: TokenAttributes, *, vocabulary: TokenVocabulary) -> int:
    return vocabulary.token_to_id(predicted_attributes_to_token(attributes))


def degree_to_attribute_id(degree: int) -> int:
    attribute_id = degree - MIN_DEGREE
    if not 0 <= attribute_id < DEGREE_ATTRIBUTE_COUNT:
        raise ValueError(f"degree must be in [{MIN_DEGREE}, {MAX_DEGREE}]")

    return attribute_id


def attribute_id_to_degree(attribute_id: int) -> int:
    _validate_attribute_id(attribute_id, attribute_name="degree_id", attribute_count=DEGREE_ATTRIBUTE_COUNT)
    return MIN_DEGREE + attribute_id


def accidental_to_attribute_id(accidental: int) -> int:
    attribute_id = accidental - MIN_ACCIDENTAL
    if not 0 <= attribute_id < ACCIDENTAL_ATTRIBUTE_COUNT:
        raise ValueError(f"accidental must be in [{MIN_ACCIDENTAL}, {MAX_ACCIDENTAL}]")

    return attribute_id


def attribute_id_to_accidental(attribute_id: int) -> int:
    _validate_attribute_id(
        attribute_id,
        attribute_name="accidental_id",
        attribute_count=ACCIDENTAL_ATTRIBUTE_COUNT,
    )
    return MIN_ACCIDENTAL + attribute_id


def octave_offset_to_attribute_id(octave_offset: int) -> int:
    attribute_id = octave_offset - MIN_OCTAVE_OFFSET
    if not 0 <= attribute_id < OCTAVE_OFFSET_ATTRIBUTE_COUNT:
        raise ValueError(f"octave_offset must be in [{MIN_OCTAVE_OFFSET}, {MAX_OCTAVE_OFFSET}]")

    return attribute_id


def attribute_id_to_octave_offset(attribute_id: int) -> int:
    _validate_attribute_id(
        attribute_id,
        attribute_name="octave_offset_id",
        attribute_count=OCTAVE_OFFSET_ATTRIBUTE_COUNT,
    )
    return MIN_OCTAVE_OFFSET + attribute_id


def hand_to_attribute_id(hand: Hand) -> int:
    return _HAND_ID_BY_HAND[hand]


def attribute_id_to_hand(attribute_id: int) -> Hand:
    _validate_attribute_id(attribute_id, attribute_name="hand_id", attribute_count=HAND_ATTRIBUTE_COUNT)
    return _HAND_BY_ID[attribute_id]


def _token_kind_id(kind_id: int) -> TokenKindId:
    try:
        return TokenKindId(kind_id)
    except ValueError as error:
        raise ValueError(f"unknown token kind id: {kind_id}") from error


def _duration_id(duration_id: int) -> int:
    if duration_id < MIN_DURATION_ID:
        raise ValueError(f"duration_id must be >= {MIN_DURATION_ID}")

    return duration_id


def _validate_attribute_id(attribute_id: int, *, attribute_name: str, attribute_count: int) -> None:
    if not 0 <= attribute_id < attribute_count:
        raise ValueError(f"{attribute_name} must be in [0, {attribute_count - 1}]")


def _validate_inactive_attributes_absent(attributes: TokenAttributes) -> None:
    kind_id = _token_kind_id(attributes.kind_id)
    match kind_id:
        case TokenKindId.NOTE:
            _require_absent(attributes.hand_id, attribute_name="hand_id")
        case TokenKindId.REST | TokenKindId.HOLD:
            _require_absent(attributes.degree_id, attribute_name="degree_id")
            _require_absent(attributes.accidental_id, attribute_name="accidental_id")
            _require_absent(attributes.octave_offset_id, attribute_name="octave_offset_id")
            _require_absent(attributes.hand_id, attribute_name="hand_id")
        case TokenKindId.HAND:
            _require_absent(attributes.degree_id, attribute_name="degree_id")
            _require_absent(attributes.accidental_id, attribute_name="accidental_id")
            _require_absent(attributes.octave_offset_id, attribute_name="octave_offset_id")
            _require_absent(attributes.duration_id, attribute_name="duration_id")
        case TokenKindId.BAR | TokenKindId.END | TokenKindId.JOIN_WITH_PREVIOUS | TokenKindId.START:
            _require_absent(attributes.degree_id, attribute_name="degree_id")
            _require_absent(attributes.accidental_id, attribute_name="accidental_id")
            _require_absent(attributes.octave_offset_id, attribute_name="octave_offset_id")
            _require_absent(attributes.duration_id, attribute_name="duration_id")
            _require_absent(attributes.hand_id, attribute_name="hand_id")


def _require_absent(attribute_id: int, *, attribute_name: str) -> None:
    if attribute_id != ABSENT_ATTRIBUTE_ID:
        raise ValueError(f"{attribute_name} must be absent")
