from musak_model.data.schema import Segment
from musak_model.tokens.schema import (
    DOUBLED_DURATIONS,
    HALVED_DURATIONS,
    MAX_OCTAVE_OFFSET,
    MIN_OCTAVE_OFFSET,
    BarToken,
    DurationClass,
    EndToken,
    NoteToken,
    RestToken,
    Token,
)


def shift_register(
    segment: Segment,
    *,
    offset: int,
) -> Segment:
    shifted_right = _shift_tokens(segment.right_hand_tokens, offset=offset)
    shifted_left = _shift_tokens(segment.left_hand_tokens, offset=offset)
    return segment.model_copy(update={"right_hand_tokens": shifted_right, "left_hand_tokens": shifted_left})


def halve_durations(segment: Segment) -> Segment:
    return _remap_durations(segment, duration_map=HALVED_DURATIONS)


def double_durations(segment: Segment) -> Segment:
    return _remap_durations(segment, duration_map=DOUBLED_DURATIONS)


def _shift_tokens(
    tokens: list[Token],
    *,
    offset: int,
) -> list[Token]:
    return [_shift_token(token, offset=offset) for token in tokens]


def _shift_token(
    token: Token,
    *,
    offset: int,
) -> Token:
    if isinstance(token, NoteToken):
        new_offset = token.octave_offset + offset
        if not MIN_OCTAVE_OFFSET <= new_offset <= MAX_OCTAVE_OFFSET:
            raise ValueError(
                f"octave offset {new_offset} after shift by {offset} is out of range "
                f"[{MIN_OCTAVE_OFFSET}, {MAX_OCTAVE_OFFSET}]"
            )

        return token.model_copy(update={"octave_offset": new_offset})

    if isinstance(token, (RestToken, BarToken, EndToken)):
        return token

    raise ValueError(f"unexpected token type: {type(token)}")


def _remap_durations(
    segment: Segment,
    *,
    duration_map: dict[DurationClass, DurationClass],
) -> Segment:
    remapped_right = _remap_tokens(segment.right_hand_tokens, duration_map=duration_map)
    remapped_left = _remap_tokens(segment.left_hand_tokens, duration_map=duration_map)
    return segment.model_copy(update={"right_hand_tokens": remapped_right, "left_hand_tokens": remapped_left})


def _remap_tokens(
    tokens: list[Token],
    *,
    duration_map: dict[DurationClass, DurationClass],
) -> list[Token]:
    return [_remap_token_duration(token, duration_map=duration_map) for token in tokens]


def _remap_token_duration(
    token: Token,
    *,
    duration_map: dict[DurationClass, DurationClass],
) -> Token:
    if isinstance(token, NoteToken):
        if token.duration not in duration_map:
            raise ValueError(f"duration {token.duration.value} has no mapping in the given duration map")
        return token.model_copy(update={"duration": duration_map[token.duration]})

    if isinstance(token, RestToken):
        if token.duration not in duration_map:
            raise ValueError(f"duration {token.duration.value} has no mapping in the given duration map")
        return token.model_copy(update={"duration": duration_map[token.duration]})

    if isinstance(token, (BarToken, EndToken)):
        return token

    raise ValueError(f"unexpected token type: {type(token)}")
