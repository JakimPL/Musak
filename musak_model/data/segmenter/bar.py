from fractions import Fraction

from musak_model.data.converter import PitchDegreeRegisterError, pitch_to_degree
from musak_model.data.schema import (
    ParsedBar,
    ParsedChord,
    ParsedEvent,
    ParsedNote,
    ParsedRest,
    ParsedScore,
    PitchDegree,
    SegmentIneligibilityReason,
    TieType,
)
from musak_model.data.segmenter.duration import exact_duration, exact_duration_id
from musak_model.data.segmenter.errors import TokenizationIneligibilityError
from musak_model.data.segmenter.events import event_pitches, event_sort_key
from musak_model.data.segmenter.types import BarNormalization, TieState, TimedTokenGroup
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HoldToken, NoteToken, RestToken, Token


def bar_measure_duration(bar: ParsedBar) -> Fraction:
    if bar.measure_duration is not None and bar.measure_duration > 0:
        return bar.measure_duration

    return Fraction(bar.time_numerator, bar.time_denominator)


def paired_bar_measure_duration(right_bar: ParsedBar, left_bar: ParsedBar) -> Fraction:
    parsed_durations = tuple(
        duration
        for duration in (right_bar.measure_duration, left_bar.measure_duration)
        if duration is not None and duration > 0
    )
    if parsed_durations:
        return max(parsed_durations)

    return max(bar_measure_duration(right_bar), bar_measure_duration(left_bar))


def normalize_bar_events(
    *,
    bar: ParsedBar,
    bar_index: int,
    hand: Hand,
    score: ParsedScore,
    duration_vocabulary: DurationVocabulary,
    measure_duration: Fraction,
    tie_state: TieState | None = None,
) -> BarNormalization:
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
    groups: list[TimedTokenGroup] = []
    current_tie_state = tie_state

    for event in sorted(bar.events, key=event_sort_key):
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
                TimedTokenGroup(
                    bar_index=bar_index,
                    offset=cursor,
                    hand=hand,
                    tokens=[
                        RestToken(
                            duration_id=exact_duration_id(
                                rest_duration,
                                vocabulary=duration_vocabulary,
                            )
                        )
                    ],
                )
            )

        event_tokens, current_tie_state = tokenize_event(
            event,
            score=score,
            hand=hand,
            duration_vocabulary=duration_vocabulary,
            tie_state=current_tie_state,
        )
        groups.append(
            TimedTokenGroup(
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
            TimedTokenGroup(
                bar_index=bar_index,
                offset=cursor,
                hand=hand,
                tokens=[
                    RestToken(
                        duration_id=exact_duration_id(
                            measure_duration - cursor,
                            vocabulary=duration_vocabulary,
                        )
                    )
                ],
            )
        )

    return BarNormalization(groups=groups, tie_state=current_tie_state)


def tokenize_bar(
    bar: ParsedBar,
    *,
    score: ParsedScore,
    hand: Hand,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    normalized = normalize_bar_events(
        bar=bar,
        bar_index=0,
        score=score,
        hand=hand,
        duration_vocabulary=duration_vocabulary,
        measure_duration=bar_measure_duration(bar),
    )
    return tokens_from_bar_groups(
        normalized.groups,
    )


def tokenize_event(
    event: ParsedEvent,
    *,
    score: ParsedScore,
    hand: Hand,
    duration_vocabulary: DurationVocabulary,
    tie_state: TieState | None,
) -> tuple[list[Token], TieState | None]:
    match event:
        case ParsedNote():
            return _tokenize_pitched_event(
                event,
                tokens=[
                    note_to_token(
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
                        duration_id=exact_duration_id(
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
                tokens=chord_to_tokens(
                    event,
                    score=score,
                    hand=hand,
                    duration_vocabulary=duration_vocabulary,
                ),
                duration_vocabulary=duration_vocabulary,
                tie_state=tie_state,
            )


def note_to_token(
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
    duration_id = exact_duration_id(event.duration, vocabulary=duration_vocabulary)
    return NoteToken(
        degree=pitch_degree.degree,
        accidental=pitch_degree.accidental,
        octave_offset=pitch_degree.octave_offset,
        duration_id=duration_id,
    )


def chord_to_tokens(
    event: ParsedChord,
    *,
    score: ParsedScore,
    hand: Hand,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    sorted_pitches = sorted(event.midi_pitches)
    tokens: list[Token] = []
    for midi_pitch in sorted_pitches:
        note_token = note_to_token(
            ParsedNote(midi_pitch=midi_pitch, duration=event.duration, beat_offset=event.beat_offset),
            score=score,
            hand=hand,
            duration_vocabulary=duration_vocabulary,
        )
        tokens.append(note_token)

    return tokens


def tokens_from_bar_groups(groups: list[TimedTokenGroup]) -> list[Token]:
    tokens: list[Token] = []
    for group in groups:
        tokens.extend(group.tokens)

    return tokens


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
    for event in sorted(bar.events, key=event_sort_key):
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
            quantized_cursor += exact_duration(
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

        for midi_pitch in event_pitches(event):
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

        quantized_cursor += exact_duration(
            event.duration,
            vocabulary=duration_vocabulary,
            bar_index=bar_index,
            hand=hand,
            context="event",
        )
        cursor = event.beat_offset + event.duration

    if cursor < measure_duration:
        exact_duration(
            measure_duration - cursor,
            vocabulary=duration_vocabulary,
            bar_index=bar_index,
            hand=hand,
            context="trailing rest",
        )


def _tokenize_pitched_event(
    event: ParsedNote | ParsedChord,
    *,
    tokens: list[Token],
    duration_vocabulary: DurationVocabulary,
    tie_state: TieState | None,
) -> tuple[list[Token], TieState | None]:
    tie_type = event.tie_type
    event_tie_state = TieState(midi_pitches=event_pitches(event))

    if tie_type == TieType.PARTIAL:
        raise TokenizationIneligibilityError(
            "partial chord ties are not supported",
            reason=SegmentIneligibilityReason.PARTIAL_CHORD_TIE,
        )

    if tie_type in {TieType.CONTINUE, TieType.STOP}:
        _raise_for_tie_state_mismatch(tie_state=tie_state, event_tie_state=event_tie_state)
        hold_tokens: list[Token] = [
            HoldToken(duration_id=exact_duration_id(event.duration, vocabulary=duration_vocabulary)),
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


def _raise_for_tie_state_mismatch(*, tie_state: TieState | None, event_tie_state: TieState) -> None:
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


def _pitch_to_degree_for_tokenization(
    event: ParsedNote,
    *,
    score: ParsedScore,
    hand: Hand,
) -> PitchDegree:
    try:
        return pitch_to_degree(
            event.midi_pitch,
            scale_root=score.scale_root,
            key_fifths=score.key_fifths,
            scale_type=score.scale_type,
            hand=hand,
        )
    except PitchDegreeRegisterError as exception:
        raise TokenizationIneligibilityError(
            (
                f"pitch {event.midi_pitch} is outside supported {hand.value} hand register "
                f"for key {score.scale_root} {score.scale_type.value}"
            ),
            reason=SegmentIneligibilityReason.REGISTER_OUT_OF_RANGE,
        ) from exception
