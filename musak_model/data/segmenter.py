from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

from pydantic import ValidationError

from musak_model.data.config import SegmentationConfig
from musak_model.data.converter import pitch_to_degree
from musak_model.data.quantizer import quantize_duration
from musak_model.data.schema import (
    ParsedBar,
    ParsedChord,
    ParsedEvent,
    ParsedNote,
    ParsedRest,
    ParsedScore,
    Segment,
    SegmentIneligibilityReason,
    SegmentMetadata,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    ScaleType,
    Token,
)

_BAR_TOKEN: BarToken = BarToken()
_END_TOKEN: EndToken = EndToken()
_JOIN_WITH_PREVIOUS_TOKEN: JoinWithPreviousToken = JoinWithPreviousToken()
_TOKENIZATION_ERRORS: tuple[type[Exception], ...] = (ValueError, ValidationError)


class TokenizationIneligibilityError(ValueError):
    def __init__(self, message: str, *, reason: SegmentIneligibilityReason) -> None:
        super().__init__(message)
        self.reason = reason


class _TimedTokenGroup(NamedTuple):
    bar_index: int
    offset: Fraction
    hand: Hand
    tokens: list[Token]


class _BarTokenization(NamedTuple):
    tokens: list[Token]
    ineligibility_reasons: frozenset[SegmentIneligibilityReason] = frozenset()


def segment_score(
    score: ParsedScore,
    source_file: Path,
    *,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    difficulty_level: int | None = None,
) -> list[Segment]:
    right_hand_tokens = _tokenize_hand_safely(
        score.right_hand_bars,
        score=score,
        hand=Hand.RIGHT,
        scale_type=scale_type,
        duration_vocabulary=duration_vocabulary,
    )
    left_hand_tokens = _tokenize_hand_safely(
        score.left_hand_bars,
        score=score,
        hand=Hand.LEFT,
        scale_type=scale_type,
        duration_vocabulary=duration_vocabulary,
    )
    unified_tokens = _tokenize_unified_stream_safely(
        score=score,
        scale_type=scale_type,
        duration_vocabulary=duration_vocabulary,
    )

    return _create_windows(
        right_hand_tokens=right_hand_tokens,
        left_hand_tokens=left_hand_tokens,
        unified_tokens=unified_tokens,
        score=score,
        source_file=source_file,
        scale_type=scale_type,
        segmentation=segmentation,
        difficulty_level=difficulty_level,
    )


def _tokenize_hand_safely(
    bars: list[ParsedBar],
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[_BarTokenization]:
    tokenized_bars: list[_BarTokenization] = []
    for index, bar in enumerate(bars):
        try:
            tokens = _tokens_from_bar_groups(
                _normalize_bar_events(
                    bar=bar,
                    bar_index=index,
                    hand=hand,
                    score=score,
                    scale_type=scale_type,
                    duration_vocabulary=duration_vocabulary,
                    measure_duration=_bar_measure_duration(bar),
                )
            )
            tokenized_bars.append(_BarTokenization(tokens=tokens))
        except _TOKENIZATION_ERRORS as exception:
            tokenized_bars.append(
                _BarTokenization(
                    tokens=[],
                    ineligibility_reasons=_ineligibility_reasons_for_exception(exception),
                )
            )

    return tokenized_bars


def _tokenize_hand(
    bars: list[ParsedBar],
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[list[Token]]:
    return [
        _tokens_from_bar_groups(
            _normalize_bar_events(
                bar=bar,
                bar_index=index,
                hand=hand,
                score=score,
                scale_type=scale_type,
                duration_vocabulary=duration_vocabulary,
                measure_duration=_bar_measure_duration(bar),
            )
        )
        for index, bar in enumerate(bars)
    ]


def _tokenize_unified_stream_safely(
    *,
    score: ParsedScore,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[_BarTokenization]:
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    tokenized_bars: list[_BarTokenization] = []

    for bar_index in range(total_bars):
        try:
            right_groups = _normalize_bar_events(
                bar=score.right_hand_bars[bar_index],
                bar_index=bar_index,
                hand=Hand.RIGHT,
                score=score,
                scale_type=scale_type,
                duration_vocabulary=duration_vocabulary,
                measure_duration=_bar_measure_duration(score.right_hand_bars[bar_index]),
            )
            left_groups = _normalize_bar_events(
                bar=score.left_hand_bars[bar_index],
                bar_index=bar_index,
                hand=Hand.LEFT,
                score=score,
                scale_type=scale_type,
                duration_vocabulary=duration_vocabulary,
                measure_duration=_bar_measure_duration(score.left_hand_bars[bar_index]),
            )
            tokenized_bars.append(_BarTokenization(tokens=_merge_hand_groups(right_groups + left_groups)))
        except _TOKENIZATION_ERRORS as exception:
            tokenized_bars.append(
                _BarTokenization(
                    tokens=[],
                    ineligibility_reasons=_ineligibility_reasons_for_exception(exception),
                )
            )

    return tokenized_bars


def _tokenize_unified_stream(
    *,
    score: ParsedScore,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[list[Token]]:
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    tokenized_bars: list[list[Token]] = []

    for bar_index in range(total_bars):
        right_groups = _normalize_bar_events(
            bar=score.right_hand_bars[bar_index],
            bar_index=bar_index,
            hand=Hand.RIGHT,
            score=score,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
            measure_duration=_bar_measure_duration(score.right_hand_bars[bar_index]),
        )
        left_groups = _normalize_bar_events(
            bar=score.left_hand_bars[bar_index],
            bar_index=bar_index,
            hand=Hand.LEFT,
            score=score,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
            measure_duration=_bar_measure_duration(score.left_hand_bars[bar_index]),
        )
        tokenized_bars.append(_merge_hand_groups(right_groups + left_groups))

    return tokenized_bars


def _normalize_bar_events(
    *,
    bar: ParsedBar,
    bar_index: int,
    hand: Hand,
    score: ParsedScore,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
    measure_duration: Fraction,
) -> list[_TimedTokenGroup]:
    _raise_for_ambiguous_simultaneous_durations(bar=bar, bar_index=bar_index, hand=hand)
    _raise_for_quantization_grid_integrity(
        bar=bar,
        bar_index=bar_index,
        hand=hand,
        duration_vocabulary=duration_vocabulary,
        measure_duration=measure_duration,
    )
    cursor = Fraction(0, 1)
    groups: list[_TimedTokenGroup] = []

    for event in sorted(bar.events, key=_event_sort_key):
        if event.beat_offset < cursor:
            raise ValueError(
                f"overlapping events in {hand.value} hand at bar {bar_index}: "
                f"event starts at {event.beat_offset}, previous event ends at {cursor}"
            )

        if event.beat_offset > cursor:
            rest_duration = event.beat_offset - cursor
            groups.append(
                _TimedTokenGroup(
                    bar_index=bar_index,
                    offset=cursor,
                    hand=hand,
                    tokens=[RestToken(duration_id=_exact_duration_id(rest_duration, vocabulary=duration_vocabulary))],
                )
            )

        groups.append(
            _TimedTokenGroup(
                bar_index=bar_index,
                offset=event.beat_offset,
                hand=hand,
                tokens=_tokenize_event(
                    event,
                    score=score,
                    hand=hand,
                    scale_type=scale_type,
                    duration_vocabulary=duration_vocabulary,
                ),
            )
        )
        cursor = event.beat_offset + event.duration

    if cursor > measure_duration:
        raise ValueError(
            f"{hand.value} hand bar {bar_index} exceeds measure duration {measure_duration}: ends at {cursor}"
        )

    if cursor < measure_duration:
        groups.append(
            _TimedTokenGroup(
                bar_index=bar_index,
                offset=cursor,
                hand=hand,
                tokens=[
                    RestToken(duration_id=_exact_duration_id(measure_duration - cursor, vocabulary=duration_vocabulary))
                ],
            )
        )

    return groups


def _event_sort_key(event: ParsedEvent) -> tuple[Fraction, int]:
    return event.beat_offset, _lowest_pitch(event)


def _raise_for_ambiguous_simultaneous_durations(*, bar: ParsedBar, bar_index: int, hand: Hand) -> None:
    durations_by_offset: dict[Fraction, set[Fraction]] = {}
    for event in bar.events:
        if isinstance(event, ParsedNote | ParsedChord):
            durations_by_offset.setdefault(event.beat_offset, set()).add(event.duration)

    for beat_offset, durations in durations_by_offset.items():
        if len(durations) > 1:
            raise TokenizationIneligibilityError(
                (
                    f"ambiguous simultaneous note durations in {hand.value} hand at bar {bar_index}, "
                    f"offset {beat_offset}: {sorted(durations)}"
                ),
                reason=SegmentIneligibilityReason.AMBIGUOUS_SIMULTANEOUS_DURATION,
            )


def _raise_for_quantization_grid_integrity(
    *,
    bar: ParsedBar,
    bar_index: int,
    hand: Hand,
    duration_vocabulary: DurationVocabulary,
    measure_duration: Fraction,
) -> None:
    cursor = Fraction(0)
    quantized_cursor = Fraction(0)
    seen_pitch_onsets: dict[tuple[Fraction, int], Fraction] = {}
    for event in sorted(bar.events, key=_event_sort_key):
        if event.beat_offset < cursor:
            raise ValueError(
                f"overlapping events in {hand.value} hand at bar {bar_index}: "
                f"event starts at {event.beat_offset}, previous event ends at {cursor}"
            )

        if event.beat_offset > cursor:
            rest_duration = event.beat_offset - cursor
            quantized_cursor += _exact_duration(
                rest_duration,
                vocabulary=duration_vocabulary,
                bar_index=bar_index,
                hand=hand,
                context="rest",
            )

        if quantized_cursor != event.beat_offset:
            raise TokenizationIneligibilityError(
                (
                    f"quantization would warp {hand.value} hand time grid at bar {bar_index}: "
                    f"event starts at {event.beat_offset}, quantized cursor is {quantized_cursor}"
                ),
                reason=SegmentIneligibilityReason.QUANTIZATION_ERROR,
            )

        for midi_pitch in _event_pitches(event):
            collision_key = (quantized_cursor, midi_pitch)
            previous_offset = seen_pitch_onsets.get(collision_key)
            if previous_offset is not None and previous_offset != event.beat_offset:
                raise TokenizationIneligibilityError(
                    (
                        f"quantization collision in {hand.value} hand at bar {bar_index}: "
                        f"pitch {midi_pitch} maps offsets {previous_offset} and {event.beat_offset} "
                        f"to {quantized_cursor}"
                    ),
                    reason=SegmentIneligibilityReason.QUANTIZATION_COLLISION,
                )
            seen_pitch_onsets[collision_key] = event.beat_offset

        quantized_cursor += _exact_duration(
            event.duration,
            vocabulary=duration_vocabulary,
            bar_index=bar_index,
            hand=hand,
            context="event",
        )
        cursor = event.beat_offset + event.duration

    if cursor < measure_duration:
        _exact_duration(
            measure_duration - cursor,
            vocabulary=duration_vocabulary,
            bar_index=bar_index,
            hand=hand,
            context="trailing rest",
        )


def _lowest_pitch(event: ParsedEvent) -> int:
    if isinstance(event, ParsedNote):
        return event.midi_pitch

    if isinstance(event, ParsedChord):
        return min(event.midi_pitches)

    return -1


def _event_pitches(event: ParsedEvent) -> tuple[int, ...]:
    if isinstance(event, ParsedNote):
        return (event.midi_pitch,)

    if isinstance(event, ParsedChord):
        return tuple(event.midi_pitches)

    return ()


def _tokens_from_bar_groups(groups: list[_TimedTokenGroup]) -> list[Token]:
    tokens: list[Token] = []
    for group in groups:
        tokens.extend(group.tokens)

    return tokens


def _merge_hand_groups(groups: list[_TimedTokenGroup]) -> list[Token]:
    tokens: list[Token] = []
    active_hand: Hand | None = None
    previous_note_onset: tuple[int, Fraction] | None = None

    for group in sorted(groups, key=_group_sort_key):
        if active_hand != group.hand:
            tokens.append(HandToken(hand=group.hand))
            active_hand = group.hand

        for token in group.tokens:
            tokens.append(token)
            if isinstance(token, NoteToken):
                current_onset = (group.bar_index, group.offset)
                if previous_note_onset == current_onset:
                    tokens.append(_JOIN_WITH_PREVIOUS_TOKEN)
                previous_note_onset = current_onset

    return tokens


def _group_sort_key(group: _TimedTokenGroup) -> tuple[Fraction, int, int]:
    return group.offset, _hand_sort_index(group.hand), _group_lowest_pitch(group)


def _hand_sort_index(hand: Hand) -> int:
    return 0 if hand == Hand.RIGHT else 1


def _group_lowest_pitch(group: _TimedTokenGroup) -> int:
    note_tokens = [index for index, token in enumerate(group.tokens) if isinstance(token, NoteToken)]
    return note_tokens[0] if note_tokens else -1


def _tokenize_bar(
    bar: ParsedBar,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    return _tokens_from_bar_groups(
        _normalize_bar_events(
            bar=bar,
            bar_index=0,
            score=score,
            hand=hand,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
            measure_duration=_bar_measure_duration(bar),
        )
    )


def _tokenize_event(
    event: ParsedEvent,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    if isinstance(event, ParsedNote):
        return [
            _note_to_token(
                event,
                score=score,
                hand=hand,
                scale_type=scale_type,
                duration_vocabulary=duration_vocabulary,
            )
        ]

    if isinstance(event, ParsedRest):
        return [RestToken(duration_id=_exact_duration_id(event.duration, vocabulary=duration_vocabulary))]

    if isinstance(event, ParsedChord):
        return _chord_to_tokens(
            event,
            score=score,
            hand=hand,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
        )

    raise ValueError(f"unexpected event type: {type(event)}")


def _note_to_token(
    event: ParsedNote,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> NoteToken:
    pitch_degree = pitch_to_degree(
        event.midi_pitch,
        key_root=score.key_root,
        key_fifths=score.key_fifths,
        scale_type=scale_type,
        hand=hand,
    )
    duration_id = _exact_duration_id(event.duration, vocabulary=duration_vocabulary)
    return NoteToken(
        degree=pitch_degree.degree,
        accidental=pitch_degree.accidental,
        octave_offset=pitch_degree.octave_offset,
        duration_id=duration_id,
    )


def _chord_to_tokens(
    event: ParsedChord,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    sorted_pitches = sorted(event.midi_pitches)
    tokens: list[Token] = []
    for midi_pitch in sorted_pitches:
        note_token = _note_to_token(
            ParsedNote(midi_pitch=midi_pitch, duration=event.duration, beat_offset=event.beat_offset),
            score=score,
            hand=hand,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
        )
        tokens.append(note_token)

    return tokens


def _create_windows(
    right_hand_tokens: list[_BarTokenization],
    left_hand_tokens: list[_BarTokenization],
    unified_tokens: list[_BarTokenization],
    score: ParsedScore,
    source_file: Path,
    *,
    scale_type: ScaleType,
    segmentation: SegmentationConfig,
    difficulty_level: int | None,
) -> list[Segment]:
    total_bars = min(len(right_hand_tokens), len(left_hand_tokens), len(unified_tokens))
    segments: list[Segment] = []

    for start in range(0, total_bars - segmentation.window_bars + 1, segmentation.stride_bars):
        end = start + segmentation.window_bars
        right_window_bars = right_hand_tokens[start:end]
        left_window_bars = left_hand_tokens[start:end]
        unified_window_bars = unified_tokens[start:end]
        right_window = _flatten_bars([bar.tokens for bar in right_window_bars])
        left_window = _flatten_bars([bar.tokens for bar in left_window_bars])
        unified_window = _flatten_bars([bar.tokens for bar in unified_window_bars])
        first_bar = score.right_hand_bars[start]
        ineligibility_reasons = _merge_ineligibility_reasons(
            _segment_ineligibility_reasons(
                score=score,
                start=start,
                end=end,
            ),
            *(bar.ineligibility_reasons for bar in right_window_bars),
            *(bar.ineligibility_reasons for bar in left_window_bars),
            *(bar.ineligibility_reasons for bar in unified_window_bars),
        )

        segments.append(
            Segment(
                tokens=unified_window,
                right_hand_tokens=right_window,
                left_hand_tokens=left_window,
                metadata=SegmentMetadata(
                    key_root=score.key_root,
                    scale_type=scale_type,
                    time_numerator=first_bar.time_numerator,
                    time_denominator=first_bar.time_denominator,
                    bar_count=segmentation.window_bars,
                    window_start_bar=start,
                    source_file=source_file,
                    difficulty_level=difficulty_level,
                    eligible_for_training=not ineligibility_reasons,
                    ineligibility_reasons=ineligibility_reasons,
                ),
            )
        )

    return segments


def _bar_measure_duration(bar: ParsedBar) -> Fraction:
    return Fraction(bar.time_numerator, bar.time_denominator)


def _segment_ineligibility_reasons(
    *,
    score: ParsedScore,
    start: int,
    end: int,
) -> frozenset[SegmentIneligibilityReason]:
    first_bar = score.right_hand_bars[start]
    first_time_signature = (first_bar.time_numerator, first_bar.time_denominator)
    first_key_fifths = first_bar.key_fifths
    reasons: set[SegmentIneligibilityReason] = set()

    window_bars = score.right_hand_bars[start:end] + score.left_hand_bars[start:end]
    if any((bar.time_numerator, bar.time_denominator) != first_time_signature for bar in window_bars):
        reasons.add(SegmentIneligibilityReason.MIXED_TIME_SIGNATURE)

    if any(bar.key_fifths != first_key_fifths for bar in window_bars):
        reasons.add(SegmentIneligibilityReason.KEY_SIGNATURE_CHANGE)

    return frozenset(reasons)


def _merge_ineligibility_reasons(
    *reason_groups: frozenset[SegmentIneligibilityReason],
) -> frozenset[SegmentIneligibilityReason]:
    return frozenset().union(*reason_groups)


def _flatten_bars(bar_token_lists: list[list[Token]]) -> list[Token]:
    tokens: list[Token] = []
    for bar_tokens in bar_token_lists:
        tokens.extend(bar_tokens)
        tokens.append(_BAR_TOKEN)

    tokens.append(_END_TOKEN)
    return tokens


def _ineligibility_reasons_for_exception(exception: Exception) -> frozenset[SegmentIneligibilityReason]:
    if isinstance(exception, TokenizationIneligibilityError):
        return frozenset({exception.reason})

    return frozenset({SegmentIneligibilityReason.TOKENIZATION_ERROR})


def _exact_duration_id(duration: Fraction, *, vocabulary: DurationVocabulary) -> int:
    quantized = quantize_duration(duration, vocabulary=vocabulary)
    if not quantized.exact:
        raise TokenizationIneligibilityError(
            f"unsupported duration {duration}: closest supported duration is {quantized.quantized}",
            reason=SegmentIneligibilityReason.QUANTIZATION_ERROR,
        )

    return quantized.duration_id


def _exact_duration(
    duration: Fraction,
    *,
    vocabulary: DurationVocabulary,
    bar_index: int,
    hand: Hand,
    context: str,
) -> Fraction:
    quantized = quantize_duration(duration, vocabulary=vocabulary)
    if not quantized.exact:
        raise TokenizationIneligibilityError(
            (
                f"unsupported {context} duration in {hand.value} hand at bar {bar_index}: "
                f"{duration} would quantize to {quantized.quantized}"
            ),
            reason=SegmentIneligibilityReason.QUANTIZATION_ERROR,
        )

    return quantized.quantized
