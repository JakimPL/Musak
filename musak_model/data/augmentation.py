from fractions import Fraction

from musak_model.data.schema import Segment
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


def shift_register(
    segment: Segment,
    *,
    offset: int,
) -> Segment:
    return segment.model_copy(update={"tokens": _shift_tokens(segment.tokens, offset=offset)})


def halve_durations(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> Segment:
    return _remap_durations(segment, duration_vocabulary=duration_vocabulary, factor=Fraction(1, 2))


def double_durations(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> Segment:
    return _remap_durations(segment, duration_vocabulary=duration_vocabulary, factor=Fraction(2, 1))


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

    if isinstance(token, (RestToken, HoldToken, HandToken, JoinWithPreviousToken, BarToken, StartToken, EndToken)):
        return token

    raise ValueError(f"unexpected token type: {type(token)}")


def _remap_durations(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    factor: Fraction,
) -> Segment:
    tokens = _remap_tokens(
        segment.tokens,
        duration_vocabulary=duration_vocabulary,
        factor=factor,
    )
    _raise_for_invalid_bar_timing(
        tokens,
        duration_vocabulary=duration_vocabulary,
        measure_duration=Fraction(segment.time_numerator, segment.time_denominator),
    )
    return segment.model_copy(update={"tokens": tokens})


def _remap_tokens(
    tokens: list[Token],
    *,
    duration_vocabulary: DurationVocabulary,
    factor: Fraction,
) -> list[Token]:
    return [
        _remap_token_duration(
            token,
            duration_vocabulary=duration_vocabulary,
            factor=factor,
        )
        for token in tokens
    ]


def _remap_token_duration(
    token: Token,
    *,
    duration_vocabulary: DurationVocabulary,
    factor: Fraction,
) -> Token:
    if isinstance(token, NoteToken):
        remapped_duration_id = _remap_duration_id(
            token.duration_id,
            duration_vocabulary=duration_vocabulary,
            factor=factor,
        )
        return token.model_copy(update={"duration_id": remapped_duration_id})

    if isinstance(token, RestToken | HoldToken):
        remapped_duration_id = _remap_duration_id(
            token.duration_id,
            duration_vocabulary=duration_vocabulary,
            factor=factor,
        )
        return token.model_copy(update={"duration_id": remapped_duration_id})

    if isinstance(token, (HandToken, JoinWithPreviousToken, BarToken, StartToken, EndToken)):
        return token

    raise ValueError(f"unexpected token type: {type(token)}")


def _remap_duration_id(
    duration_id: int,
    *,
    duration_vocabulary: DurationVocabulary,
    factor: Fraction,
) -> int:
    source_fraction = duration_vocabulary.id_to_fraction(duration_id)
    target_fraction = source_fraction * factor
    candidate_duration_id, candidate_fraction = duration_vocabulary.find_closest(target_fraction)
    if candidate_fraction != target_fraction:
        raise ValueError(f"duration {source_fraction} has no exact mapping for factor {factor}")

    return candidate_duration_id


def _raise_for_invalid_bar_timing(
    tokens: list[Token],
    *,
    duration_vocabulary: DurationVocabulary,
    measure_duration: Fraction,
) -> None:
    cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}
    active_hand = Hand.RIGHT

    for token in tokens:
        if isinstance(token, HandToken):
            active_hand = token.hand
            continue

        if isinstance(token, BarToken):
            _raise_for_bar_overflow(cursors, measure_duration=measure_duration)
            cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}
            continue

        if isinstance(token, EndToken):
            _raise_for_bar_overflow(cursors, measure_duration=measure_duration)
            break

        if isinstance(token, StartToken):
            continue

        if isinstance(token, RestToken | NoteToken | HoldToken):
            cursors[active_hand] += duration_vocabulary.id_to_fraction(token.duration_id)
            continue

        if isinstance(token, JoinWithPreviousToken):
            continue

        raise ValueError(f"unexpected token type: {type(token)}")


def _raise_for_bar_overflow(cursors: dict[Hand, Fraction], *, measure_duration: Fraction) -> None:
    for hand, cursor in cursors.items():
        if cursor > measure_duration:
            raise ValueError(f"{hand.value} hand duration {cursor} exceeds measure duration {measure_duration}")
