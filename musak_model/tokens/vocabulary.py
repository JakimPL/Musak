from typing import Final

from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    DEGREE_COUNT,
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

_BAR_TOKEN: Final[BarToken] = BarToken()
_START_TOKEN: Final[StartToken] = StartToken()
_END_TOKEN: Final[EndToken] = EndToken()
_JOIN_WITH_PREVIOUS_TOKEN: Final[JoinWithPreviousToken] = JoinWithPreviousToken()

_ACCIDENTAL_COUNT: Final[int] = MAX_ACCIDENTAL - MIN_ACCIDENTAL + 1
_OCTAVE_OFFSET_COUNT: Final[int] = MAX_OCTAVE_OFFSET - MIN_OCTAVE_OFFSET + 1


class TokenVocabulary:
    def __init__(self, duration_vocabulary: DurationVocabulary) -> None:
        self._duration_vocabulary = duration_vocabulary
        self._duration_count = duration_vocabulary.vocabulary_size()
        self._note_count = DEGREE_COUNT * _ACCIDENTAL_COUNT * _OCTAVE_OFFSET_COUNT * self._duration_count
        self._rest_count = self._duration_count
        self._hold_count = self._duration_count
        self._bar_token_id = self._note_count + self._rest_count + self._hold_count
        self._end_token_id = self._bar_token_id + 1
        self._right_hand_token_id = self._end_token_id + 1
        self._left_hand_token_id = self._right_hand_token_id + 1
        self._join_with_previous_token_id = self._left_hand_token_id + 1
        self._start_token_id = self._join_with_previous_token_id + 1
        self._vocab_size = self._start_token_id + 1

    @property
    def vocabulary_size(self) -> int:
        return self._vocab_size

    @property
    def duration_vocabulary(self) -> DurationVocabulary:
        return self._duration_vocabulary

    @property
    def bar_token_id(self) -> int:
        return self._bar_token_id

    @property
    def end_token_id(self) -> int:
        return self._end_token_id

    @property
    def right_hand_token_id(self) -> int:
        return self._right_hand_token_id

    @property
    def left_hand_token_id(self) -> int:
        return self._left_hand_token_id

    @property
    def join_with_previous_token_id(self) -> int:
        return self._join_with_previous_token_id

    @property
    def first_hold_token_id(self) -> int:
        return self._note_count + self._rest_count

    @property
    def start_token_id(self) -> int:
        return self._start_token_id

    def token_to_id(self, token: Token) -> int:
        if isinstance(token, NoteToken):
            return self._note_token_to_id(token)

        if isinstance(token, RestToken):
            return self._rest_token_to_id(token)

        if isinstance(token, HoldToken):
            return self._hold_token_to_id(token)

        if isinstance(token, BarToken):
            return self._bar_token_id

        if isinstance(token, EndToken):
            return self._end_token_id

        if isinstance(token, HandToken):
            return self._right_hand_token_id if token.hand == Hand.RIGHT else self._left_hand_token_id

        if isinstance(token, JoinWithPreviousToken):
            return self._join_with_previous_token_id

        if isinstance(token, StartToken):
            return self._start_token_id

        raise ValueError(f"unexpected token type: {type(token)}")

    def id_to_token(self, token_id: int) -> Token:
        if not 0 <= token_id < self._vocab_size:
            raise ValueError(f"token_id must be in [0, {self._vocab_size - 1}]")

        if token_id < self._note_count:
            return self._id_to_note_token(token_id)

        first_hold_token_id = self.first_hold_token_id
        if token_id < first_hold_token_id:
            duration_id = token_id - self._note_count
            return RestToken(duration_id=duration_id)

        if token_id < self._bar_token_id:
            duration_id = token_id - first_hold_token_id
            return HoldToken(duration_id=duration_id)

        if token_id == self._bar_token_id:
            return _BAR_TOKEN

        if token_id == self._end_token_id:
            return _END_TOKEN

        if token_id == self._right_hand_token_id:
            return HandToken(hand=Hand.RIGHT)

        if token_id == self._left_hand_token_id:
            return HandToken(hand=Hand.LEFT)

        if token_id == self._join_with_previous_token_id:
            return _JOIN_WITH_PREVIOUS_TOKEN

        return _START_TOKEN

    def encode(self, tokens: list[Token]) -> list[int]:
        return [self.token_to_id(token) for token in tokens]

    def decode(self, token_ids: list[int]) -> list[Token]:
        return [self.id_to_token(token_id) for token_id in token_ids]

    def _note_token_to_id(self, token: NoteToken) -> int:
        _validate_duration_id(duration_id=token.duration_id, duration_count=self._duration_count)
        degree_index = token.degree - MIN_DEGREE
        accidental_index = token.accidental - MIN_ACCIDENTAL
        octave_offset_index = token.octave_offset - MIN_OCTAVE_OFFSET

        if not 0 <= degree_index < DEGREE_COUNT:
            raise ValueError(f"degree must be in [{MIN_DEGREE}, {MAX_DEGREE}]")

        if not 0 <= accidental_index < _ACCIDENTAL_COUNT:
            raise ValueError(f"accidental must be in [{MIN_ACCIDENTAL}, {MAX_ACCIDENTAL}]")

        if not 0 <= octave_offset_index < _OCTAVE_OFFSET_COUNT:
            raise ValueError(f"octave_offset must be in [{MIN_OCTAVE_OFFSET}, {MAX_OCTAVE_OFFSET}]")

        linear_index = degree_index
        linear_index = linear_index * _ACCIDENTAL_COUNT + accidental_index
        linear_index = linear_index * _OCTAVE_OFFSET_COUNT + octave_offset_index
        linear_index = linear_index * self._duration_count + token.duration_id
        return linear_index

    def _rest_token_to_id(self, token: RestToken) -> int:
        _validate_duration_id(duration_id=token.duration_id, duration_count=self._duration_count)
        return self._note_count + token.duration_id

    def _hold_token_to_id(self, token: HoldToken) -> int:
        _validate_duration_id(duration_id=token.duration_id, duration_count=self._duration_count)
        return self.first_hold_token_id + token.duration_id

    def _id_to_note_token(self, token_id: int) -> NoteToken:
        duration_id = token_id % self._duration_count
        packed_index = token_id // self._duration_count

        octave_offset_index = packed_index % _OCTAVE_OFFSET_COUNT
        packed_index //= _OCTAVE_OFFSET_COUNT

        accidental_index = packed_index % _ACCIDENTAL_COUNT
        degree_index = packed_index // _ACCIDENTAL_COUNT

        return NoteToken(
            degree=MIN_DEGREE + degree_index,
            accidental=MIN_ACCIDENTAL + accidental_index,
            octave_offset=MIN_OCTAVE_OFFSET + octave_offset_index,
            duration_id=duration_id,
        )


def _validate_duration_id(*, duration_id: int, duration_count: int) -> None:
    if not MIN_DURATION_ID <= duration_id < duration_count:
        raise ValueError(f"duration_id must be in [{MIN_DURATION_ID}, {duration_count - 1}]")


def build_default_token_vocabulary() -> TokenVocabulary:
    tokenization_config = TokenizationConfig.load()
    duration_vocabulary = DurationVocabulary(tokenization_config)
    return TokenVocabulary(duration_vocabulary)


def token_to_id(token: Token, *, vocabulary: TokenVocabulary) -> int:
    return vocabulary.token_to_id(token)


def id_to_token(token_id: int, *, vocabulary: TokenVocabulary) -> Token:
    return vocabulary.id_to_token(token_id)


def encode(tokens: list[Token], *, vocabulary: TokenVocabulary) -> list[int]:
    return vocabulary.encode(tokens)


def decode(token_ids: list[int], *, vocabulary: TokenVocabulary) -> list[Token]:
    return vocabulary.decode(token_ids)
