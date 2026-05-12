from typing import Final

from musak_model.tokens.schema import (
    MAX_ACCIDENTAL,
    MAX_DEGREE,
    MAX_OCTAVE_OFFSET,
    MIN_ACCIDENTAL,
    MIN_DEGREE,
    MIN_OCTAVE_OFFSET,
    BarToken,
    DurationClass,
    EndToken,
    NoteToken,
    RestToken,
    Token,
)

_BAR_TOKEN: Final[BarToken] = BarToken()
_END_TOKEN: Final[EndToken] = EndToken()


def _enumerate_note_tokens() -> list[NoteToken]:
    return [
        NoteToken(
            degree=degree,
            accidental=accidental,
            octave_offset=octave_offset,
            duration=duration,
        )
        for degree in range(MIN_DEGREE, MAX_DEGREE + 1)
        for accidental in range(MIN_ACCIDENTAL, MAX_ACCIDENTAL + 1)
        for octave_offset in range(MIN_OCTAVE_OFFSET, MAX_OCTAVE_OFFSET + 1)
        for duration in DurationClass
    ]


def _enumerate_rest_tokens() -> list[RestToken]:
    return [RestToken(duration=duration) for duration in DurationClass]


def _build_token_to_id() -> dict[Token, int]:
    tokens: list[Token] = [
        *_enumerate_note_tokens(),
        *_enumerate_rest_tokens(),
        _BAR_TOKEN,
        _END_TOKEN,
    ]
    return {token: index for index, token in enumerate(tokens)}


_TOKEN_TO_ID: Final[dict[Token, int]] = _build_token_to_id()
_ID_TO_TOKEN: Final[dict[int, Token]] = {index: token for token, index in _TOKEN_TO_ID.items()}

VOCAB_SIZE: Final[int] = len(_TOKEN_TO_ID)

BAR_TOKEN_ID: Final[int] = _TOKEN_TO_ID[_BAR_TOKEN]
END_TOKEN_ID: Final[int] = _TOKEN_TO_ID[_END_TOKEN]


def token_to_id(token: Token) -> int:
    return _TOKEN_TO_ID[token]


def id_to_token(token_id: int) -> Token:
    return _ID_TO_TOKEN[token_id]


def encode(tokens: list[Token]) -> list[int]:
    return [token_to_id(token) for token in tokens]


def decode(token_ids: list[int]) -> list[Token]:
    return [id_to_token(token_id) for token_id in token_ids]
