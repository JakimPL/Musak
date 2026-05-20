from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

from musak_model.data.cleaning import is_silent_bar_pair
from musak_model.data.config import SegmentationConfig
from musak_model.data.converter import PitchDegreeRegisterError, pitch_to_degree
from musak_model.data.quantizer import quantize_duration
from musak_model.data.schema import (
    ParsedBar,
    ParsedChord,
    ParsedEvent,
    ParsedNote,
    ParsedRest,
    ParsedScore,
    PitchDegree,
    Segment,
    SegmentIneligibilityReason,
    SegmentMetadata,
    TieType,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    Token,
)

_BAR_TOKEN: BarToken = BarToken()
_END_TOKEN: EndToken = EndToken()
_JOIN_WITH_PREVIOUS_TOKEN: JoinWithPreviousToken = JoinWithPreviousToken()


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


class _TieState(NamedTuple):
    midi_pitches: tuple[int, ...]


class _BarNormalization(NamedTuple):
    groups: list[_TimedTokenGroup]
    tie_state: _TieState | None


def segment_score(
    score: ParsedScore,
    source_file: Path,
    *,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    difficulty_level: int | None = None,
) -> list[Segment]:
    unified_tokens = _tokenize_unified_stream_safely(
        score=score,
        duration_vocabulary=duration_vocabulary,
    )

    return _create_windows(
        unified_tokens=unified_tokens,
        score=score,
        source_file=source_file,
        segmentation=segmentation,
        difficulty_level=difficulty_level,
    )


def _tokenize_hand_safely(
    bars: list[ParsedBar],
    *,
    score: ParsedScore,
    hand: Hand,
    duration_vocabulary: DurationVocabulary,
) -> list[_BarTokenization]:
    tokenized_bars: list[_BarTokenization] = []
    tie_state: _TieState | None = None
    for index, bar in enumerate(bars):
        try:
            normalized = _normalize_bar_events(
                bar=bar,
                bar_index=index,
                hand=hand,
                score=score,
                duration_vocabulary=duration_vocabulary,
                measure_duration=_bar_measure_duration(bar),
                tie_state=tie_state,
            )
            tie_state = normalized.tie_state
            tokens = _tokens_from_bar_groups(normalized.groups)
            tokenized_bars.append(_BarTokenization(tokens=tokens))
        except TokenizationIneligibilityError as exception:
            tie_state = None
            tokenized_bars.append(
                _BarTokenization(
                    tokens=[],
                    ineligibility_reasons=frozenset({exception.reason}),
                )
            )

    return tokenized_bars


def _tokenize_hand(
    bars: list[ParsedBar],
    *,
    score: ParsedScore,
    hand: Hand,
    duration_vocabulary: DurationVocabulary,
) -> list[list[Token]]:
    tokenized_bars: list[list[Token]] = []
    tie_state: _TieState | None = None
    for index, bar in enumerate(bars):
        normalized = _normalize_bar_events(
            bar=bar,
            bar_index=index,
            hand=hand,
            score=score,
            duration_vocabulary=duration_vocabulary,
            measure_duration=_bar_measure_duration(bar),
            tie_state=tie_state,
        )
        tie_state = normalized.tie_state
        tokenized_bars.append(_tokens_from_bar_groups(normalized.groups))

    return tokenized_bars


def _tokenize_unified_stream_safely(
    *,
    score: ParsedScore,
    duration_vocabulary: DurationVocabulary,
) -> list[_BarTokenization]:
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    tokenized_bars: list[_BarTokenization] = []
    right_tie_state: _TieState | None = None
    left_tie_state: _TieState | None = None

    for bar_index in range(total_bars):
        try:
            right_normalized = _normalize_bar_events(
                bar=score.right_hand_bars[bar_index],
                bar_index=bar_index,
                hand=Hand.RIGHT,
                score=score,
                duration_vocabulary=duration_vocabulary,
                measure_duration=_bar_measure_duration(score.right_hand_bars[bar_index]),
                tie_state=right_tie_state,
            )
            left_normalized = _normalize_bar_events(
                bar=score.left_hand_bars[bar_index],
                bar_index=bar_index,
                hand=Hand.LEFT,
                score=score,
                duration_vocabulary=duration_vocabulary,
                measure_duration=_bar_measure_duration(score.left_hand_bars[bar_index]),
                tie_state=left_tie_state,
            )
            right_tie_state = right_normalized.tie_state
            left_tie_state = left_normalized.tie_state
            tokenized_bars.append(
                _BarTokenization(tokens=_merge_hand_groups(right_normalized.groups + left_normalized.groups))
            )
        except TokenizationIneligibilityError as exception:
            right_tie_state = None
            left_tie_state = None
            tokenized_bars.append(
                _BarTokenization(
                    tokens=[],
                    ineligibility_reasons=frozenset({exception.reason}),
                )
            )

    return tokenized_bars


def _tokenize_unified_stream(
    *,
    score: ParsedScore,
    duration_vocabulary: DurationVocabulary,
) -> list[list[Token]]:
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    tokenized_bars: list[list[Token]] = []
    right_tie_state: _TieState | None = None
    left_tie_state: _TieState | None = None

    for bar_index in range(total_bars):
        right_normalized = _normalize_bar_events(
            bar=score.right_hand_bars[bar_index],
            bar_index=bar_index,
            hand=Hand.RIGHT,
            score=score,
            duration_vocabulary=duration_vocabulary,
            measure_duration=_bar_measure_duration(score.right_hand_bars[bar_index]),
            tie_state=right_tie_state,
        )
        left_normalized = _normalize_bar_events(
            bar=score.left_hand_bars[bar_index],
            bar_index=bar_index,
            hand=Hand.LEFT,
            score=score,
            duration_vocabulary=duration_vocabulary,
            measure_duration=_bar_measure_duration(score.left_hand_bars[bar_index]),
            tie_state=left_tie_state,
        )
        right_tie_state = right_normalized.tie_state
        left_tie_state = left_normalized.tie_state
        tokenized_bars.append(_merge_hand_groups(right_normalized.groups + left_normalized.groups))

    return tokenized_bars


def _normalize_bar_events(
    *,
    bar: ParsedBar,
    bar_index: int,
    hand: Hand,
    score: ParsedScore,
    duration_vocabulary: DurationVocabulary,
    measure_duration: Fraction,
    tie_state: _TieState | None = None,
) -> _BarNormalization:
    _raise_for_ambiguous_simultaneous_durations(
        bar=bar,
        bar_index=bar_index,
        hand=hand,
    )
    _raise_for_quantization_grid_integrity(
        bar=bar,
        bar_index=bar_index,
        hand=hand,
        duration_vocabulary=duration_vocabulary,
        measure_duration=measure_duration,
    )
    cursor = Fraction(0, 1)
    groups: list[_TimedTokenGroup] = []
    current_tie_state = tie_state

    for event in sorted(bar.events, key=_event_sort_key):
        if event.beat_offset < cursor:
            raise TokenizationIneligibilityError(
                (
                    f"overlapping events in {hand.value} hand at bar {bar_index}: "
                    f"event starts at {event.beat_offset}, previous event ends at {cursor}"
                ),
                reason=SegmentIneligibilityReason.OVERLAPPING_EVENTS,
            )

        if event.beat_offset > cursor:
            if current_tie_state is not None:
                raise TokenizationIneligibilityError(
                    f"open tie in {hand.value} hand at bar {bar_index} has a gap before {event.beat_offset}",
                    reason=SegmentIneligibilityReason.TIE_MISMATCH,
                )

            rest_duration = event.beat_offset - cursor
            groups.append(
                _TimedTokenGroup(
                    bar_index=bar_index,
                    offset=cursor,
                    hand=hand,
                    tokens=[
                        RestToken(
                            duration_id=_exact_duration_id(
                                rest_duration,
                                vocabulary=duration_vocabulary,
                            )
                        )
                    ],
                )
            )

        event_tokens, current_tie_state = _tokenize_event(
            event,
            score=score,
            hand=hand,
            duration_vocabulary=duration_vocabulary,
            tie_state=current_tie_state,
        )
        groups.append(
            _TimedTokenGroup(
                bar_index=bar_index,
                offset=event.beat_offset,
                hand=hand,
                tokens=event_tokens,
            )
        )
        cursor = event.beat_offset + event.duration

    if cursor > measure_duration:
        raise TokenizationIneligibilityError(
            f"{hand.value} hand bar {bar_index} exceeds measure duration {measure_duration}: ends at {cursor}",
            reason=SegmentIneligibilityReason.BAR_DURATION_OVERFLOW,
        )

    if cursor < measure_duration:
        groups.append(
            _TimedTokenGroup(
                bar_index=bar_index,
                offset=cursor,
                hand=hand,
                tokens=[
                    RestToken(
                        duration_id=_exact_duration_id(
                            measure_duration - cursor,
                            vocabulary=duration_vocabulary,
                        )
                    )
                ],
            )
        )

    return _BarNormalization(groups=groups, tie_state=current_tie_state)


def _event_sort_key(event: ParsedEvent) -> tuple[Fraction, int]:
    return event.beat_offset, _lowest_pitch(event)


def _raise_for_ambiguous_simultaneous_durations(
    *,
    bar: ParsedBar,
    bar_index: int,
    hand: Hand,
) -> None:
    durations_by_offset: dict[Fraction, set[Fraction]] = {}
    for event in bar.events:
        match event:
            case ParsedNote() | ParsedChord():
                durations_by_offset.setdefault(event.beat_offset, set()).add(event.duration)
            case ParsedRest():
                continue

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
            raise TokenizationIneligibilityError(
                (
                    f"overlapping events in {hand.value} hand at bar {bar_index}: "
                    f"event starts at {event.beat_offset}, previous event ends at {cursor}"
                ),
                reason=SegmentIneligibilityReason.OVERLAPPING_EVENTS,
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
    match event:
        case ParsedNote():
            return event.midi_pitch
        case ParsedChord():
            return min(event.midi_pitches)
        case ParsedRest():
            return -1


def _event_pitches(event: ParsedEvent) -> tuple[int, ...]:
    match event:
        case ParsedNote():
            return (event.midi_pitch,)
        case ParsedChord():
            return tuple(sorted(event.midi_pitches))
        case ParsedRest():
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
            match token:
                case NoteToken():
                    current_onset = (group.bar_index, group.offset)
                    if previous_note_onset == current_onset:
                        tokens.append(_JOIN_WITH_PREVIOUS_TOKEN)
                    previous_note_onset = current_onset
                case HoldToken() | RestToken() | HandToken() | BarToken() | EndToken() | JoinWithPreviousToken():
                    continue

    return tokens


def _group_sort_key(group: _TimedTokenGroup) -> tuple[Fraction, int, int]:
    return group.offset, _hand_sort_index(group.hand), _group_lowest_pitch(group)


def _hand_sort_index(hand: Hand) -> int:
    match hand:
        case Hand.RIGHT:
            return 0
        case Hand.LEFT:
            return 1


def _group_lowest_pitch(group: _TimedTokenGroup) -> int:
    for index, token in enumerate(group.tokens):
        match token:
            case NoteToken():
                return index
            case HoldToken() | RestToken() | HandToken() | BarToken() | EndToken() | JoinWithPreviousToken():
                continue

    return -1


def _tokenize_bar(
    bar: ParsedBar,
    *,
    score: ParsedScore,
    hand: Hand,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    normalized = _normalize_bar_events(
        bar=bar,
        bar_index=0,
        score=score,
        hand=hand,
        duration_vocabulary=duration_vocabulary,
        measure_duration=_bar_measure_duration(bar),
    )
    return _tokens_from_bar_groups(
        normalized.groups,
    )


def _tokenize_event(
    event: ParsedEvent,
    *,
    score: ParsedScore,
    hand: Hand,
    duration_vocabulary: DurationVocabulary,
    tie_state: _TieState | None,
) -> tuple[list[Token], _TieState | None]:
    match event:
        case ParsedNote():
            return _tokenize_pitched_event(
                event,
                tokens=[
                    _note_to_token(
                        event,
                        score=score,
                        hand=hand,
                        duration_vocabulary=duration_vocabulary,
                    )
                ],
                duration_vocabulary=duration_vocabulary,
                tie_state=tie_state,
            )
        case ParsedRest():
            if tie_state is not None:
                raise TokenizationIneligibilityError(
                    "rest cannot continue an open tie",
                    reason=SegmentIneligibilityReason.TIE_MISMATCH,
                )

            return (
                [
                    RestToken(
                        duration_id=_exact_duration_id(
                            event.duration,
                            vocabulary=duration_vocabulary,
                        )
                    )
                ],
                None,
            )
        case ParsedChord():
            return _tokenize_pitched_event(
                event,
                tokens=_chord_to_tokens(
                    event,
                    score=score,
                    hand=hand,
                    duration_vocabulary=duration_vocabulary,
                ),
                duration_vocabulary=duration_vocabulary,
                tie_state=tie_state,
            )


def _tokenize_pitched_event(
    event: ParsedNote | ParsedChord,
    *,
    tokens: list[Token],
    duration_vocabulary: DurationVocabulary,
    tie_state: _TieState | None,
) -> tuple[list[Token], _TieState | None]:
    tie_type = event.tie_type
    event_tie_state = _TieState(midi_pitches=_event_pitches(event))

    if tie_type == TieType.PARTIAL:
        raise TokenizationIneligibilityError(
            "partial chord ties are not supported",
            reason=SegmentIneligibilityReason.PARTIAL_CHORD_TIE,
        )

    if tie_type in {TieType.CONTINUE, TieType.STOP}:
        _raise_for_tie_state_mismatch(tie_state=tie_state, event_tie_state=event_tie_state)
        hold_tokens: list[Token] = [
            HoldToken(duration_id=_exact_duration_id(event.duration, vocabulary=duration_vocabulary)),
        ]
        return hold_tokens, event_tie_state if tie_type == TieType.CONTINUE else None

    if tie_state is not None:
        raise TokenizationIneligibilityError(
            "open tie was not continued by a matching event",
            reason=SegmentIneligibilityReason.TIE_MISMATCH,
        )

    if tie_type == TieType.START:
        return tokens, event_tie_state

    return tokens, None


def _raise_for_tie_state_mismatch(*, tie_state: _TieState | None, event_tie_state: _TieState) -> None:
    if tie_state is None:
        raise TokenizationIneligibilityError(
            "tie continuation has no matching open tie",
            reason=SegmentIneligibilityReason.TIE_MISMATCH,
        )

    if tie_state.midi_pitches != event_tie_state.midi_pitches:
        raise TokenizationIneligibilityError(
            f"tie continuation pitches {event_tie_state.midi_pitches} do not match open tie {tie_state.midi_pitches}",
            reason=SegmentIneligibilityReason.TIE_MISMATCH,
        )


def _note_to_token(
    event: ParsedNote,
    *,
    score: ParsedScore,
    hand: Hand,
    duration_vocabulary: DurationVocabulary,
) -> NoteToken:
    pitch_degree = _pitch_to_degree_for_tokenization(
        event,
        score=score,
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
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    sorted_pitches = sorted(event.midi_pitches)
    tokens: list[Token] = []
    for midi_pitch in sorted_pitches:
        note_token = _note_to_token(
            ParsedNote(midi_pitch=midi_pitch, duration=event.duration, beat_offset=event.beat_offset),
            score=score,
            hand=hand,
            duration_vocabulary=duration_vocabulary,
        )
        tokens.append(note_token)

    return tokens


def _create_windows(
    unified_tokens: list[_BarTokenization],
    score: ParsedScore,
    source_file: Path,
    *,
    segmentation: SegmentationConfig,
    difficulty_level: int | None,
) -> list[Segment]:
    total_bars = len(unified_tokens)
    segments: list[Segment] = []

    for start in range(0, total_bars - segmentation.window_bars + 1, segmentation.stride_bars):
        end = start + segmentation.window_bars
        unified_window_bars = unified_tokens[start:end]
        unified_window = _flatten_bars([bar.tokens for bar in unified_window_bars])
        first_bar = score.right_hand_bars[start]
        ineligibility_reasons = _merge_ineligibility_reasons(
            _segment_ineligibility_reasons(
                score=score,
                start=start,
                end=end,
            ),
            _window_token_ineligibility_reasons(
                first_bar_tokens=unified_window_bars[0].tokens,
                window_start_bar=start,
            ),
            *(bar.ineligibility_reasons for bar in unified_window_bars),
        )

        segments.append(
            Segment(
                tokens=unified_window,
                metadata=SegmentMetadata(
                    key_root=score.key_root,
                    scale_type=score.scale_type,
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

    if _has_silent_edge_bar(score=score, start=start, end=end):
        reasons.add(SegmentIneligibilityReason.SILENT_EDGE_BAR)

    return frozenset(reasons)


def _has_silent_edge_bar(*, score: ParsedScore, start: int, end: int) -> bool:
    first_index = start
    last_index = end - 1
    return is_silent_bar_pair(
        score.right_hand_bars[first_index],
        score.left_hand_bars[first_index],
    ) or is_silent_bar_pair(
        score.right_hand_bars[last_index],
        score.left_hand_bars[last_index],
    )


def _window_token_ineligibility_reasons(
    *,
    first_bar_tokens: list[Token],
    window_start_bar: int,
) -> frozenset[SegmentIneligibilityReason]:
    if window_start_bar == 0:
        return frozenset()

    active_hand = Hand.RIGHT
    seen_same_hand_attack = {Hand.RIGHT: False, Hand.LEFT: False}
    for token in first_bar_tokens:
        if isinstance(token, HandToken):
            active_hand = token.hand
            continue

        if isinstance(token, HoldToken) and not seen_same_hand_attack[active_hand]:
            return frozenset({SegmentIneligibilityReason.TIE_CONTINUATION_AT_WINDOW_START})

        if isinstance(token, NoteToken):
            seen_same_hand_attack[active_hand] = True

    return frozenset()


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


def _pitch_to_degree_for_tokenization(
    event: ParsedNote,
    *,
    score: ParsedScore,
    hand: Hand,
) -> PitchDegree:
    try:
        return pitch_to_degree(
            event.midi_pitch,
            key_root=score.key_root,
            key_fifths=score.key_fifths,
            scale_type=score.scale_type,
            hand=hand,
        )
    except PitchDegreeRegisterError as exception:
        raise TokenizationIneligibilityError(
            (
                f"pitch {event.midi_pitch} is outside supported {hand.value} hand register "
                f"for key {score.key_root} {score.scale_type.value}"
            ),
            reason=SegmentIneligibilityReason.REGISTER_OUT_OF_RANGE,
        ) from exception


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
