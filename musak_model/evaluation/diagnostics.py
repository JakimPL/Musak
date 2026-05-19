from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.data.schema import Segment
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
    StartToken,
)

_HANDS: Final[tuple[Hand, Hand]] = (Hand.RIGHT, Hand.LEFT)


class SegmentDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    right_silence_fraction: float = Field(ge=0)
    left_silence_fraction: float = Field(ge=0)
    both_hands_silence_fraction: float = Field(ge=0)
    both_hands_active_fraction: float = Field(ge=0)
    right_only_active_fraction: float = Field(ge=0)
    left_only_active_fraction: float = Field(ge=0)
    longest_right_silence_beats: float = Field(ge=0)
    longest_left_silence_beats: float = Field(ge=0)
    longest_both_hands_silence_beats: float = Field(ge=0)
    right_note_onsets_per_bar: float = Field(ge=0)
    left_note_onsets_per_bar: float = Field(ge=0)
    hand_activity_balance: float = Field(ge=0, le=1)
    empty_score: bool
    one_hand_only: bool
    note_token_fraction: float = Field(ge=0, le=1)
    rest_token_fraction: float = Field(ge=0, le=1)
    hold_token_fraction: float = Field(ge=0, le=1)

    def manifest_values(self) -> dict[str, float | bool]:
        return self.model_dump()


def diagnose_segment(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> SegmentDiagnostics:
    events = _segment_activity_events(segment, duration_vocabulary=duration_vocabulary)
    total_duration = _segment_duration(segment, events)
    active_intervals = {
        hand: _merged_intervals((event.start, event.end) for event in events if event.hand == hand) for hand in _HANDS
    }
    state_durations = _hand_state_durations(active_intervals, total_duration=total_duration)
    right_active_duration = state_durations.right_only_active_duration + state_durations.both_hands_active_duration
    left_active_duration = state_durations.left_only_active_duration + state_durations.both_hands_active_duration
    bar_count = max(segment.bar_count, 1)

    return SegmentDiagnostics(
        right_silence_fraction=_duration_fraction(total_duration - right_active_duration, total_duration),
        left_silence_fraction=_duration_fraction(total_duration - left_active_duration, total_duration),
        both_hands_silence_fraction=_duration_fraction(state_durations.both_hands_silence_duration, total_duration),
        both_hands_active_fraction=_duration_fraction(state_durations.both_hands_active_duration, total_duration),
        right_only_active_fraction=_duration_fraction(state_durations.right_only_active_duration, total_duration),
        left_only_active_fraction=_duration_fraction(state_durations.left_only_active_duration, total_duration),
        longest_right_silence_beats=_beats(
            _longest_silence(active_intervals[Hand.RIGHT], total_duration=total_duration),
            denominator=segment.time_denominator,
        ),
        longest_left_silence_beats=_beats(
            _longest_silence(active_intervals[Hand.LEFT], total_duration=total_duration),
            denominator=segment.time_denominator,
        ),
        longest_both_hands_silence_beats=_beats(
            state_durations.longest_both_hands_silence,
            denominator=segment.time_denominator,
        ),
        right_note_onsets_per_bar=_onsets_per_bar(events, hand=Hand.RIGHT, bar_count=bar_count),
        left_note_onsets_per_bar=_onsets_per_bar(events, hand=Hand.LEFT, bar_count=bar_count),
        hand_activity_balance=_activity_balance(right_active_duration, left_active_duration),
        empty_score=right_active_duration == 0 and left_active_duration == 0,
        one_hand_only=(right_active_duration == 0) != (left_active_duration == 0),
        note_token_fraction=_token_fraction(segment, NoteToken),
        rest_token_fraction=_token_fraction(segment, RestToken),
        hold_token_fraction=_token_fraction(segment, HoldToken),
    )


class _HandStateDurations(BaseModel):
    model_config = ConfigDict(frozen=True)

    both_hands_silence_duration: Fraction
    both_hands_active_duration: Fraction
    right_only_active_duration: Fraction
    left_only_active_duration: Fraction
    longest_both_hands_silence: Fraction


@dataclass(frozen=True)
class _ActivityEvent:
    hand: Hand
    start: Fraction
    duration: Fraction

    @property
    def end(self) -> Fraction:
        return self.start + self.duration


def _segment_activity_events(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> list[_ActivityEvent]:
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    active_hand = Hand.RIGHT
    bar_index = 0
    cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}
    last_attack_indices: dict[Hand, list[int]] = {Hand.RIGHT: [], Hand.LEFT: []}
    events: list[_ActivityEvent] = []

    for token in segment.tokens:
        match token:
            case HandToken(hand=hand):
                active_hand = hand
            case StartToken():
                continue
            case BarToken():
                bar_index += 1
                cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}
            case EndToken():
                break
            case RestToken(duration_id=duration_id):
                cursors[active_hand] += duration_vocabulary.id_to_fraction(duration_id)
            case HoldToken(duration_id=duration_id):
                duration = duration_vocabulary.id_to_fraction(duration_id)
                _extend_last_attack(events, event_indices=last_attack_indices[active_hand], duration=duration)
                cursors[active_hand] += duration
            case NoteToken(duration_id=duration_id):
                duration = duration_vocabulary.id_to_fraction(duration_id)
                events.append(
                    _ActivityEvent(
                        hand=active_hand,
                        start=bar_index * measure_duration + cursors[active_hand],
                        duration=duration,
                    )
                )
                cursors[active_hand] += duration
                last_attack_indices[active_hand] = [len(events) - 1]
            case JoinWithPreviousToken():
                if len(events) < 2:
                    raise ValueError("join-with-previous token needs at least two decoded notes")

                previous_event = events[-1]
                joined_start = events[-2].start
                events[-1] = _ActivityEvent(
                    hand=previous_event.hand,
                    start=joined_start,
                    duration=previous_event.duration,
                )
                last_attack_indices[active_hand] = _same_onset_event_indices(
                    events,
                    hand=active_hand,
                    start=joined_start,
                )
                cursors[active_hand] = max(
                    cursors[active_hand] - previous_event.duration,
                    events[-1].end - bar_index * measure_duration,
                )

    return events


def _extend_last_attack(events: list[_ActivityEvent], *, event_indices: list[int], duration: Fraction) -> None:
    if not event_indices:
        raise ValueError("hold token needs a previous same-hand note or chord")

    for event_index in event_indices:
        event = events[event_index]
        events[event_index] = _ActivityEvent(
            hand=event.hand,
            start=event.start,
            duration=event.duration + duration,
        )


def _same_onset_event_indices(events: list[_ActivityEvent], *, hand: Hand, start: Fraction) -> list[int]:
    return [index for index, event in enumerate(events) if event.hand == hand and event.start == start]


def _segment_duration(segment: Segment, events: list[_ActivityEvent]) -> Fraction:
    metadata_duration = segment.bar_count * Fraction(segment.time_numerator, segment.time_denominator)
    if metadata_duration > 0:
        return metadata_duration

    event_end = max((event.end for event in events), default=Fraction(0))
    return event_end


def _merged_intervals(intervals: Iterable[tuple[Fraction, Fraction]]) -> list[tuple[Fraction, Fraction]]:
    sorted_intervals = sorted(interval for interval in intervals if interval[1] > interval[0])
    merged: list[tuple[Fraction, Fraction]] = []
    for start, end in sorted_intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue

        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))

    return merged


def _hand_state_durations(
    active_intervals: dict[Hand, list[tuple[Fraction, Fraction]]],
    *,
    total_duration: Fraction,
) -> _HandStateDurations:
    if total_duration <= 0:
        return _HandStateDurations(
            both_hands_silence_duration=Fraction(0),
            both_hands_active_duration=Fraction(0),
            right_only_active_duration=Fraction(0),
            left_only_active_duration=Fraction(0),
            longest_both_hands_silence=Fraction(0),
        )

    boundaries = {Fraction(0), total_duration}
    for intervals in active_intervals.values():
        for start, end in intervals:
            boundaries.add(_clamp(start, total_duration=total_duration))
            boundaries.add(_clamp(end, total_duration=total_duration))

    both_silence = Fraction(0)
    both_active = Fraction(0)
    right_only = Fraction(0)
    left_only = Fraction(0)
    longest_both_silence = Fraction(0)
    sorted_boundaries = sorted(boundaries)
    for start, end in zip(sorted_boundaries, sorted_boundaries[1:]):
        if start == end:
            continue

        duration = end - start
        right_active = _contains_interval(active_intervals[Hand.RIGHT], start=start, end=end)
        left_active = _contains_interval(active_intervals[Hand.LEFT], start=start, end=end)
        if right_active and left_active:
            both_active += duration
        elif right_active:
            right_only += duration
        elif left_active:
            left_only += duration
        else:
            both_silence += duration
            longest_both_silence = max(longest_both_silence, duration)

    return _HandStateDurations(
        both_hands_silence_duration=both_silence,
        both_hands_active_duration=both_active,
        right_only_active_duration=right_only,
        left_only_active_duration=left_only,
        longest_both_hands_silence=longest_both_silence,
    )


def _longest_silence(intervals: list[tuple[Fraction, Fraction]], *, total_duration: Fraction) -> Fraction:
    if total_duration <= 0:
        return Fraction(0)

    longest = Fraction(0)
    cursor = Fraction(0)
    for start, end in intervals:
        clamped_start = _clamp(start, total_duration=total_duration)
        clamped_end = _clamp(end, total_duration=total_duration)
        longest = max(longest, clamped_start - cursor)
        cursor = max(cursor, clamped_end)

    return max(longest, total_duration - cursor)


def _contains_interval(intervals: list[tuple[Fraction, Fraction]], *, start: Fraction, end: Fraction) -> bool:
    return any(interval_start <= start and end <= interval_end for interval_start, interval_end in intervals)


def _clamp(value: Fraction, *, total_duration: Fraction) -> Fraction:
    return min(max(value, Fraction(0)), total_duration)


def _duration_fraction(duration: Fraction, total_duration: Fraction) -> float:
    if total_duration <= 0:
        return 0.0

    return float(max(Fraction(0), duration) / total_duration)


def _beats(duration: Fraction, *, denominator: int) -> float:
    return float(duration * denominator)


def _onsets_per_bar(events: list[_ActivityEvent], *, hand: Hand, bar_count: int) -> float:
    onsets = {event.start for event in events if event.hand == hand}
    return len(onsets) / bar_count


def _activity_balance(right_active_duration: Fraction, left_active_duration: Fraction) -> float:
    maximum_duration = max(right_active_duration, left_active_duration)
    if maximum_duration == 0:
        return 1.0

    return float(min(right_active_duration, left_active_duration) / maximum_duration)


def _token_fraction(segment: Segment, token_type: type[NoteToken] | type[RestToken] | type[HoldToken]) -> float:
    if not segment.tokens:
        return 0.0

    return sum(isinstance(token, token_type) for token in segment.tokens) / len(segment.tokens)
