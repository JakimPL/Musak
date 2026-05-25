from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from musak_model.data.schema import Segment
from musak_model.evaluation.diagnostics.constants import HANDS
from musak_model.evaluation.diagnostics.events import ActivityEvent
from musak_model.tokens.schema import Hand


@dataclass(frozen=True)
class HandStateDurations:
    both_hands_silence_duration: Fraction
    both_hands_active_duration: Fraction
    right_only_active_duration: Fraction
    left_only_active_duration: Fraction
    longest_both_hands_silence: Fraction


def merge_intervals(intervals: Iterable[tuple[Fraction, Fraction]]) -> list[tuple[Fraction, Fraction]]:
    sorted_intervals = sorted(interval for interval in intervals if interval[1] > interval[0])
    merged: list[tuple[Fraction, Fraction]] = []
    for start, end in sorted_intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue

        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))

    return merged


def calculate_hand_state_durations(
    active_intervals: dict[Hand, list[tuple[Fraction, Fraction]]],
    *,
    total_duration: Fraction,
) -> HandStateDurations:
    if total_duration <= 0:
        return HandStateDurations(
            both_hands_silence_duration=Fraction(0),
            both_hands_active_duration=Fraction(0),
            right_only_active_duration=Fraction(0),
            left_only_active_duration=Fraction(0),
            longest_both_hands_silence=Fraction(0),
        )

    boundaries = {Fraction(0), total_duration}
    for intervals in active_intervals.values():
        for start, end in intervals:
            boundaries.add(clamp(start, total_duration=total_duration))
            boundaries.add(clamp(end, total_duration=total_duration))

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
        right_active = contains_interval(active_intervals[Hand.RIGHT], start=start, end=end)
        left_active = contains_interval(active_intervals[Hand.LEFT], start=start, end=end)
        if right_active and left_active:
            both_active += duration
        elif right_active:
            right_only += duration
        elif left_active:
            left_only += duration
        else:
            both_silence += duration
            longest_both_silence = max(longest_both_silence, duration)

    return HandStateDurations(
        both_hands_silence_duration=both_silence,
        both_hands_active_duration=both_active,
        right_only_active_duration=right_only,
        left_only_active_duration=left_only,
        longest_both_hands_silence=longest_both_silence,
    )


def calculate_longest_silence(intervals: list[tuple[Fraction, Fraction]], *, total_duration: Fraction) -> Fraction:
    if total_duration <= 0:
        return Fraction(0)

    longest = Fraction(0)
    cursor = Fraction(0)
    for start, end in intervals:
        clamped_start = clamp(start, total_duration=total_duration)
        clamped_end = clamp(end, total_duration=total_duration)
        longest = max(longest, clamped_start - cursor)
        cursor = max(cursor, clamped_end)

    return max(longest, total_duration - cursor)


def contains_interval(intervals: list[tuple[Fraction, Fraction]], *, start: Fraction, end: Fraction) -> bool:
    return any(interval_start <= start and end <= interval_end for interval_start, interval_end in intervals)


def clamp(value: Fraction, *, total_duration: Fraction) -> Fraction:
    return min(max(value, Fraction(0)), total_duration)


def calculate_duration_fraction(duration: Fraction, total_duration: Fraction) -> float:
    if total_duration <= 0:
        return 0.0

    return float(max(Fraction(0), duration) / total_duration)


def calculate_beats(duration: Fraction, *, denominator: int) -> float:
    return float(duration * denominator)


def calculate_onsets_per_bar(events: list[ActivityEvent], *, hand: Hand, bar_count: int) -> float:
    onsets = {event.start for event in events if event.hand == hand}
    return len(onsets) / bar_count


def collect_onset_starts(events: list[ActivityEvent]) -> dict[Hand, set[Fraction]]:
    return {hand: {event.start for event in events if event.hand == hand} for hand in HANDS}


def calculate_onset_fraction(onsets: set[Fraction], *, all_onsets: set[Fraction]) -> float:
    if not all_onsets:
        return 0.0

    return len(onsets) / len(all_onsets)


def calculate_count_per_beat(count: int, *, total_beats: float) -> float:
    if total_beats <= 0:
        return 0.0

    return count / total_beats


def calculate_duration_beats(duration: Fraction | None, *, denominator: int) -> float:
    if duration is None:
        return 0.0

    return calculate_beats(duration, denominator=denominator)


def calculate_activity_balance(right_active_duration: Fraction, left_active_duration: Fraction) -> float:
    maximum_duration = max(right_active_duration, left_active_duration)
    if maximum_duration == 0:
        return 1.0

    return float(min(right_active_duration, left_active_duration) / maximum_duration)


def find_silent_bar_indices(segment: Segment, events: list[ActivityEvent]) -> set[int]:
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    if segment.bar_count <= 0 or measure_duration <= 0:
        return set()

    active_bar_indices = {
        bar_index
        for event in events
        for bar_index in find_event_bar_indices(event, measure_duration=measure_duration, bar_count=segment.bar_count)
    }
    return set(range(segment.bar_count)) - active_bar_indices


def find_event_bar_indices(event: ActivityEvent, *, measure_duration: Fraction, bar_count: int) -> range:
    first_bar = max(0, int(event.start // measure_duration))
    end_position = event.end / measure_duration
    last_bar = int(end_position) - 1 if end_position.denominator == 1 else int(end_position)
    last_bar = min(bar_count - 1, last_bar)
    return range(first_bar, last_bar + 1)


def count_silent_edge_bars(silent_bars: set[int], *, bar_count: int) -> int:
    if bar_count <= 0:
        return 0

    edge_indices = {0, bar_count - 1}
    return len(silent_bars & edge_indices)
