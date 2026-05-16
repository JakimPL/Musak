from __future__ import annotations

import re
from collections.abc import Sequence
from fractions import Fraction
from typing import Final

from musak_model.common.ratios import format_ratio
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    MAX_OCTAVE_OFFSET,
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
from musak_model.tokens.symbols import (
    ASCII_FLAT_SYMBOL,
    ASCII_SHARP_SYMBOL,
    BAR_SYMBOL,
    DURATION_CLOSE_SYMBOL,
    DURATION_OPEN_SYMBOL,
    DURATION_SEPARATOR_SYMBOL,
    END_SYMBOL,
    HOLD_SYMBOL,
    JOIN_WITH_PREVIOUS_SYMBOL,
    LEFT_HAND_SYMBOL,
    OCTAVE_DOWN_SYMBOL,
    OCTAVE_UP_SYMBOL,
    REST_SYMBOL,
    RIGHT_HAND_SYMBOL,
    START_SYMBOL,
    TEXT_FLAT_SYMBOL,
    TEXT_SHARP_SYMBOL,
)

_DURATION_PATTERN: Final[str] = (
    rf"{re.escape(DURATION_OPEN_SYMBOL)}(?P<num>\d+)"
    rf"{re.escape(DURATION_SEPARATOR_SYMBOL)}(?P<den>\d+){re.escape(DURATION_CLOSE_SYMBOL)}"
)
_ACCIDENTAL_PATTERN: Final[str] = "".join(
    re.escape(symbol) for symbol in (TEXT_SHARP_SYMBOL, TEXT_FLAT_SYMBOL, ASCII_SHARP_SYMBOL, ASCII_FLAT_SYMBOL)
)
_OCTAVE_PATTERN: Final[str] = "".join(re.escape(symbol) for symbol in (OCTAVE_UP_SYMBOL, OCTAVE_DOWN_SYMBOL))
_NOTE_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<degree>[1-7])(?P<accidental>[{_ACCIDENTAL_PATTERN}]?)(?P<octave>[{_OCTAVE_PATTERN}]\d+)?"
    rf"{_DURATION_PATTERN}$"
)
_REST_PATTERN: Final[re.Pattern[str]] = re.compile(rf"^{re.escape(REST_SYMBOL)}{_DURATION_PATTERN}$")
_HOLD_PATTERN: Final[re.Pattern[str]] = re.compile(rf"^{re.escape(HOLD_SYMBOL)}{_DURATION_PATTERN}$")

_ACCIDENTAL_VALUES: Final[dict[str, int]] = {
    "": 0,
    TEXT_SHARP_SYMBOL: 1,
    ASCII_SHARP_SYMBOL: 1,
    TEXT_FLAT_SYMBOL: -1,
    ASCII_FLAT_SYMBOL: -1,
}
_HAND_TOKENS: Final[dict[str, Hand]] = {RIGHT_HAND_SYMBOL: Hand.RIGHT, LEFT_HAND_SYMBOL: Hand.LEFT}


class TokenTextError(ValueError):
    """Base class for token text serialization and parsing errors."""


class TokenTextParseError(TokenTextError):
    """Raised when token text does not match the canonical grammar."""


class UnsupportedTokenDurationError(TokenTextParseError):
    """Raised when token text uses a duration outside the active vocabulary."""


def tokens_to_text(
    tokens: Sequence[Token],
    *,
    duration_vocabulary: DurationVocabulary,
) -> str:
    return " ".join(token.to_text(duration_vocabulary=duration_vocabulary) for token in tokens)


def tokens_from_text(
    text: str,
    *,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    stripped_text = text.strip()
    if not stripped_text:
        return []

    tokens: list[Token] = []
    for index, token_text in enumerate(stripped_text.split()):
        try:
            tokens.append(token_from_text(token_text, duration_vocabulary=duration_vocabulary))
        except UnsupportedTokenDurationError as exception:
            raise UnsupportedTokenDurationError(f"invalid token at position {index}: {exception}") from exception
        except TokenTextParseError as exception:
            raise TokenTextParseError(f"invalid token at position {index}: {exception}") from exception

    return tokens


def token_from_text(
    token_text: str,
    *,
    duration_vocabulary: DurationVocabulary,
) -> Token:
    if token_text in _HAND_TOKENS:
        return HandToken(hand=_HAND_TOKENS[token_text])

    if token_text == JOIN_WITH_PREVIOUS_SYMBOL:
        return JoinWithPreviousToken()

    if token_text == BAR_SYMBOL:
        return BarToken()

    if token_text == START_SYMBOL:
        return StartToken()

    if token_text == END_SYMBOL:
        return EndToken()

    rest_match = _REST_PATTERN.fullmatch(token_text)
    if rest_match is not None:
        return RestToken(
            duration_id=_duration_id(
                rest_match,
                token_text,
                duration_vocabulary=duration_vocabulary,
            )
        )

    hold_match = _HOLD_PATTERN.fullmatch(token_text)
    if hold_match is not None:
        return HoldToken(
            duration_id=_duration_id(
                hold_match,
                token_text,
                duration_vocabulary=duration_vocabulary,
            )
        )

    note_match = _NOTE_PATTERN.fullmatch(token_text)
    if note_match is not None:
        return NoteToken(
            degree=int(note_match.group("degree")),
            accidental=_ACCIDENTAL_VALUES[note_match.group("accidental")],
            octave_offset=_parse_octave_offset(note_match.group("octave"), token_text),
            duration_id=_duration_id(note_match, token_text, duration_vocabulary=duration_vocabulary),
        )

    raise TokenTextParseError(f"unrecognized token text: {token_text!r}")


def _parse_octave_offset(octave_text: str | None, token_text: str) -> int:
    if octave_text is None:
        return 0

    direction = octave_text[0]
    value = int(octave_text[1:])
    octave_offset = value if direction == OCTAVE_UP_SYMBOL else -value
    if not MIN_OCTAVE_OFFSET <= octave_offset <= MAX_OCTAVE_OFFSET:
        raise TokenTextParseError(
            f"octave offset in {token_text!r} must be in [{MIN_OCTAVE_OFFSET}, {MAX_OCTAVE_OFFSET}]"
        )

    return octave_offset


def _duration_id(
    match: re.Match[str],
    token_text: str,
    *,
    duration_vocabulary: DurationVocabulary,
) -> int:
    numerator = int(match.group("num"))
    denominator = int(match.group("den"))
    if numerator <= 0 or denominator <= 0:
        raise TokenTextParseError(f"duration in {token_text!r} must be positive")

    duration = Fraction(numerator, denominator)
    try:
        return duration_vocabulary.fraction_to_id(duration)
    except KeyError as exception:
        raise UnsupportedTokenDurationError(
            f"duration {format_ratio(duration, separator=DURATION_SEPARATOR_SYMBOL)} "
            f"in {token_text!r} is not supported "
            "by the active duration vocabulary"
        ) from exception
