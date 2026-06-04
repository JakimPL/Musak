from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from musak_model.data.schema import Segment
from musak_model.decoder.piano_roll import PianoRollEvent, segment_to_piano_roll_events
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand
from musak_shared.elements import PITCHES_PER_OCTAVE

_METRIC_PREFIX: Final[str] = "coherence"
_LONG_NOTE_BAR_FRACTION: Final[Fraction] = Fraction(1, 2)
_WHOLE_NOTE_DURATION: Final[Fraction] = Fraction(1)
_STEPWISE_MAX_SEMITONES: Final[int] = 2
_LARGE_LEAP_MIN_SEMITONES: Final[int] = 7
_LEAP_RECOVERY_MAX_SEMITONES: Final[int] = 5


@dataclass(frozen=True)
class _MelodicCounts:
    intervals: int
    stepwise_intervals: int
    repeated_intervals: int
    large_leaps: int
    recoveries: int
    recovery_opportunities: int
    direction_changes: int
    direction_change_opportunities: int


@dataclass(frozen=True)
class _HandDialogueCounts:
    onsets: int
    answered_onsets: int
    onset_positions: int
    synchronized_positions: int


@dataclass(frozen=True)
class _ClosureCounts:
    samples: int
    final_activity: int
    both_hands_active: int
    left_root_support: int
    right_tonic_closure: int
    long_final_note: int


def coherence_metrics(
    segments: Sequence[Segment],
    *,
    duration_vocabulary: DurationVocabulary,
    metric_prefix: str = _METRIC_PREFIX,
) -> dict[str, float]:
    note_events = 0
    long_note_events = 0
    whole_bar_note_events = 0
    whole_note_or_longer_events = 0
    samples_with_long_note = 0
    samples_with_whole_bar_note = 0
    samples_with_whole_note_or_longer = 0
    left_note_events = 0
    left_long_note_events = 0
    left_whole_bar_note_events = 0
    left_whole_note_or_longer_events = 0
    right_note_events = 0
    right_long_note_events = 0
    right_whole_bar_note_events = 0
    right_whole_note_or_longer_events = 0
    long_left_under_right_motion = 0

    melodic_counts = _MelodicCounts(0, 0, 0, 0, 0, 0, 0, 0)
    hand_dialogue_counts = _HandDialogueCounts(0, 0, 0, 0)
    closure_counts = _ClosureCounts(0, 0, 0, 0, 0, 0)

    for segment in segments:
        events = tuple(segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary))
        note_events += len(events)
        long_counts = _long_duration_counts(segment, events)
        long_note_events += long_counts.long_note_events
        whole_bar_note_events += long_counts.whole_bar_note_events
        whole_note_or_longer_events += long_counts.whole_note_or_longer_events
        samples_with_long_note += long_counts.samples_with_long_note
        samples_with_whole_bar_note += long_counts.samples_with_whole_bar_note
        samples_with_whole_note_or_longer += long_counts.samples_with_whole_note_or_longer
        left_note_events += long_counts.left_note_events
        left_long_note_events += long_counts.left_long_note_events
        left_whole_bar_note_events += long_counts.left_whole_bar_note_events
        left_whole_note_or_longer_events += long_counts.left_whole_note_or_longer_events
        right_note_events += long_counts.right_note_events
        right_long_note_events += long_counts.right_long_note_events
        right_whole_bar_note_events += long_counts.right_whole_bar_note_events
        right_whole_note_or_longer_events += long_counts.right_whole_note_or_longer_events
        long_left_under_right_motion += long_counts.long_left_under_right_motion

        melodic_counts = _add_melodic_counts(melodic_counts, _melodic_counts(events))
        hand_dialogue_counts = _add_hand_dialogue_counts(
            hand_dialogue_counts,
            _hand_dialogue_counts(segment, events),
        )
        closure_counts = _add_closure_counts(closure_counts, _closure_counts(segment, events))

    metrics: dict[str, float] = {
        f"{metric_prefix}/count/samples": float(len(segments)),
        f"{metric_prefix}/count/note_events": float(note_events),
        f"{metric_prefix}/count/melodic_intervals": float(melodic_counts.intervals),
        f"{metric_prefix}/count/large_leaps": float(melodic_counts.large_leaps),
    }
    metrics.update(
        _long_duration_metrics(
            metric_prefix,
            samples=len(segments),
            note_events=note_events,
            long_note_events=long_note_events,
            whole_bar_note_events=whole_bar_note_events,
            whole_note_or_longer_events=whole_note_or_longer_events,
            samples_with_long_note=samples_with_long_note,
            samples_with_whole_bar_note=samples_with_whole_bar_note,
            samples_with_whole_note_or_longer=samples_with_whole_note_or_longer,
            left_note_events=left_note_events,
            left_long_note_events=left_long_note_events,
            left_whole_bar_note_events=left_whole_bar_note_events,
            left_whole_note_or_longer_events=left_whole_note_or_longer_events,
            right_note_events=right_note_events,
            right_long_note_events=right_long_note_events,
            right_whole_bar_note_events=right_whole_bar_note_events,
            right_whole_note_or_longer_events=right_whole_note_or_longer_events,
            long_left_under_right_motion=long_left_under_right_motion,
        )
    )
    metrics.update(_melodic_metrics(metric_prefix, melodic_counts))
    metrics.update(_hand_dialogue_metrics(metric_prefix, hand_dialogue_counts))
    metrics.update(_closure_metrics(metric_prefix, closure_counts))
    return metrics


@dataclass(frozen=True)
class _LongDurationCounts:
    long_note_events: int
    whole_bar_note_events: int
    whole_note_or_longer_events: int
    samples_with_long_note: int
    samples_with_whole_bar_note: int
    samples_with_whole_note_or_longer: int
    left_note_events: int
    left_long_note_events: int
    left_whole_bar_note_events: int
    left_whole_note_or_longer_events: int
    right_note_events: int
    right_long_note_events: int
    right_whole_bar_note_events: int
    right_whole_note_or_longer_events: int
    long_left_under_right_motion: int


def _long_duration_counts(segment: Segment, events: tuple[PianoRollEvent, ...]) -> _LongDurationCounts:
    long_note_events = 0
    whole_bar_note_events = 0
    whole_note_or_longer_events = 0
    left_note_events = 0
    left_long_note_events = 0
    left_whole_bar_note_events = 0
    left_whole_note_or_longer_events = 0
    right_note_events = 0
    right_long_note_events = 0
    right_whole_bar_note_events = 0
    right_whole_note_or_longer_events = 0
    long_left_under_right_motion = 0

    for event in events:
        bar_duration = _bar_duration_at_position(segment, event.start)
        is_long = event.duration >= bar_duration * _LONG_NOTE_BAR_FRACTION
        is_whole_bar = event.duration >= bar_duration
        is_whole_note_or_longer = event.duration >= _WHOLE_NOTE_DURATION
        if is_long:
            long_note_events += 1
        if is_whole_bar:
            whole_bar_note_events += 1
        if is_whole_note_or_longer:
            whole_note_or_longer_events += 1

        if event.hand == Hand.LEFT:
            left_note_events += 1
            left_long_note_events += int(is_long)
            left_whole_bar_note_events += int(is_whole_bar)
            left_whole_note_or_longer_events += int(is_whole_note_or_longer)
            if is_long and _right_hand_moves_under_left_event(event, events):
                long_left_under_right_motion += 1
        elif event.hand == Hand.RIGHT:
            right_note_events += 1
            right_long_note_events += int(is_long)
            right_whole_bar_note_events += int(is_whole_bar)
            right_whole_note_or_longer_events += int(is_whole_note_or_longer)

    return _LongDurationCounts(
        long_note_events=long_note_events,
        whole_bar_note_events=whole_bar_note_events,
        whole_note_or_longer_events=whole_note_or_longer_events,
        samples_with_long_note=int(long_note_events > 0),
        samples_with_whole_bar_note=int(whole_bar_note_events > 0),
        samples_with_whole_note_or_longer=int(whole_note_or_longer_events > 0),
        left_note_events=left_note_events,
        left_long_note_events=left_long_note_events,
        left_whole_bar_note_events=left_whole_bar_note_events,
        left_whole_note_or_longer_events=left_whole_note_or_longer_events,
        right_note_events=right_note_events,
        right_long_note_events=right_long_note_events,
        right_whole_bar_note_events=right_whole_bar_note_events,
        right_whole_note_or_longer_events=right_whole_note_or_longer_events,
        long_left_under_right_motion=long_left_under_right_motion,
    )


def _right_hand_moves_under_left_event(left_event: PianoRollEvent, events: tuple[PianoRollEvent, ...]) -> bool:
    right_starts = {
        event.start for event in events if event.hand == Hand.RIGHT and left_event.start <= event.start < left_event.end
    }
    return len(right_starts) >= 2


def _melodic_counts(events: tuple[PianoRollEvent, ...]) -> _MelodicCounts:
    counts = _MelodicCounts(0, 0, 0, 0, 0, 0, 0, 0)
    for hand in Hand:
        representatives = _hand_onset_representatives(events, hand=hand)
        intervals = tuple(
            next_representative.midi_pitch - representative.midi_pitch
            for representative, next_representative in zip(representatives, representatives[1:])
        )
        counts = _add_melodic_counts(counts, _melodic_counts_from_intervals(intervals))

    return counts


def _melodic_counts_from_intervals(intervals: tuple[int, ...]) -> _MelodicCounts:
    stepwise_intervals = sum(0 < abs(interval) <= _STEPWISE_MAX_SEMITONES for interval in intervals)
    repeated_intervals = sum(interval == 0 for interval in intervals)
    large_leaps = sum(abs(interval) >= _LARGE_LEAP_MIN_SEMITONES for interval in intervals)
    recoveries = 0
    recovery_opportunities = 0
    direction_changes = 0
    direction_change_opportunities = 0
    for interval, next_interval in zip(intervals, intervals[1:]):
        if interval == 0 or next_interval == 0:
            continue

        direction_change_opportunities += 1
        if _opposite_direction(interval, next_interval):
            direction_changes += 1

        if abs(interval) >= _LARGE_LEAP_MIN_SEMITONES:
            recovery_opportunities += 1
            if _opposite_direction(interval, next_interval) and abs(next_interval) <= _LEAP_RECOVERY_MAX_SEMITONES:
                recoveries += 1

    return _MelodicCounts(
        intervals=len(intervals),
        stepwise_intervals=stepwise_intervals,
        repeated_intervals=repeated_intervals,
        large_leaps=large_leaps,
        recoveries=recoveries,
        recovery_opportunities=recovery_opportunities,
        direction_changes=direction_changes,
        direction_change_opportunities=direction_change_opportunities,
    )


def _hand_onset_representatives(
    events: tuple[PianoRollEvent, ...],
    *,
    hand: Hand,
) -> tuple[PianoRollEvent, ...]:
    events_by_start: dict[Fraction, list[PianoRollEvent]] = defaultdict(list)
    for event in events:
        if event.hand == hand:
            events_by_start[event.start].append(event)

    representatives: list[PianoRollEvent] = []
    for start in sorted(events_by_start):
        simultaneous_events = events_by_start[start]
        if hand == Hand.RIGHT:
            representatives.append(max(simultaneous_events, key=lambda event: event.midi_pitch))
        else:
            representatives.append(min(simultaneous_events, key=lambda event: event.midi_pitch))

    return tuple(representatives)


def _opposite_direction(interval: int, next_interval: int) -> bool:
    return interval > 0 > next_interval or interval < 0 < next_interval


def _hand_dialogue_counts(segment: Segment, events: tuple[PianoRollEvent, ...]) -> _HandDialogueCounts:
    right_onsets = _hand_onsets(events, hand=Hand.RIGHT)
    left_onsets = _hand_onsets(events, hand=Hand.LEFT)
    answer_window = Fraction(1, segment.time_denominator)

    right_answered = _answered_onsets(right_onsets, other_onsets=left_onsets, answer_window=answer_window)
    left_answered = _answered_onsets(left_onsets, other_onsets=right_onsets, answer_window=answer_window)
    onset_positions = frozenset({*right_onsets, *left_onsets})
    synchronized_positions = frozenset(right_onsets).intersection(left_onsets)
    return _HandDialogueCounts(
        onsets=len(right_onsets) + len(left_onsets),
        answered_onsets=right_answered + left_answered,
        onset_positions=len(onset_positions),
        synchronized_positions=len(synchronized_positions),
    )


def _hand_onsets(events: tuple[PianoRollEvent, ...], *, hand: Hand) -> tuple[Fraction, ...]:
    return tuple(sorted({event.start for event in events if event.hand == hand}))


def _answered_onsets(
    onsets: tuple[Fraction, ...],
    *,
    other_onsets: tuple[Fraction, ...],
    answer_window: Fraction,
) -> int:
    return sum(
        _has_answer(
            onset,
            other_onsets=other_onsets,
            answer_window=answer_window,
        )
        for onset in onsets
    )


def _has_answer(
    onset: Fraction,
    *,
    other_onsets: tuple[Fraction, ...],
    answer_window: Fraction,
) -> bool:
    return any(onset < other_onset <= onset + answer_window for other_onset in other_onsets)


def _closure_counts(segment: Segment, events: tuple[PianoRollEvent, ...]) -> _ClosureCounts:
    if segment.bar_count == 0:
        return _ClosureCounts(0, 0, 0, 0, 0, 0)

    final_position = _segment_duration(segment)
    final_sounding_events = tuple(event for event in events if event.start < final_position <= event.end)
    if not final_sounding_events:
        return _ClosureCounts(1, 0, 0, 0, 0, 0)

    left_final_events = tuple(event for event in final_sounding_events if event.hand == Hand.LEFT)
    right_final_events = tuple(event for event in final_sounding_events if event.hand == Hand.RIGHT)
    final_bar_duration = _bar_duration_at_position(segment, final_position)
    tonic_pitch_class = segment.scale_root % PITCHES_PER_OCTAVE
    highest_right_event = max(right_final_events, key=lambda event: event.midi_pitch) if right_final_events else None
    return _ClosureCounts(
        samples=1,
        final_activity=1,
        both_hands_active=int(bool(left_final_events) and bool(right_final_events)),
        left_root_support=int(
            any(event.midi_pitch % PITCHES_PER_OCTAVE == tonic_pitch_class for event in left_final_events)
        ),
        right_tonic_closure=int(
            highest_right_event is not None and highest_right_event.midi_pitch % PITCHES_PER_OCTAVE == tonic_pitch_class
        ),
        long_final_note=int(
            any(event.duration >= final_bar_duration * _LONG_NOTE_BAR_FRACTION for event in final_sounding_events)
        ),
    )


def _segment_duration(segment: Segment) -> Fraction:
    if segment.metadata.bar_durations is None:
        return segment.bar_count * _default_bar_duration(segment)

    return sum(segment.metadata.bar_durations[: segment.bar_count], start=Fraction(0))


def _bar_duration_at_position(segment: Segment, position: Fraction) -> Fraction:
    if segment.metadata.bar_durations is None:
        return _default_bar_duration(segment)

    elapsed = Fraction(0)
    for bar_duration in segment.metadata.bar_durations[: segment.bar_count]:
        bar_end = elapsed + bar_duration
        if position < bar_end:
            return bar_duration
        elapsed = bar_end

    if segment.metadata.bar_durations and segment.bar_count > 0:
        return segment.metadata.bar_durations[min(segment.bar_count, len(segment.metadata.bar_durations)) - 1]

    return _default_bar_duration(segment)


def _default_bar_duration(segment: Segment) -> Fraction:
    return Fraction(segment.time_numerator, segment.time_denominator)


def _long_duration_metrics(
    metric_prefix: str,
    *,
    samples: int,
    note_events: int,
    long_note_events: int,
    whole_bar_note_events: int,
    whole_note_or_longer_events: int,
    samples_with_long_note: int,
    samples_with_whole_bar_note: int,
    samples_with_whole_note_or_longer: int,
    left_note_events: int,
    left_long_note_events: int,
    left_whole_bar_note_events: int,
    left_whole_note_or_longer_events: int,
    right_note_events: int,
    right_long_note_events: int,
    right_whole_bar_note_events: int,
    right_whole_note_or_longer_events: int,
    long_left_under_right_motion: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        f"{metric_prefix}/count/long_note_events": float(long_note_events),
        f"{metric_prefix}/count/whole_bar_note_events": float(whole_bar_note_events),
        f"{metric_prefix}/count/whole_note_or_longer_events": float(whole_note_or_longer_events),
    }
    if samples > 0:
        metrics[f"{metric_prefix}/rate/samples_with_long_note"] = samples_with_long_note / samples
        metrics[f"{metric_prefix}/rate/samples_with_whole_bar_note"] = samples_with_whole_bar_note / samples
        metrics[f"{metric_prefix}/rate/samples_with_whole_note_or_longer"] = samples_with_whole_note_or_longer / samples
    if note_events > 0:
        metrics[f"{metric_prefix}/rate/long_note_events"] = long_note_events / note_events
        metrics[f"{metric_prefix}/rate/whole_bar_note_events"] = whole_bar_note_events / note_events
        metrics[f"{metric_prefix}/rate/whole_note_or_longer_events"] = whole_note_or_longer_events / note_events
    if left_note_events > 0:
        metrics[f"{metric_prefix}/rate/left_long_note_events"] = left_long_note_events / left_note_events
        metrics[f"{metric_prefix}/rate/left_whole_bar_note_events"] = left_whole_bar_note_events / left_note_events
        metrics[f"{metric_prefix}/rate/left_whole_note_or_longer_events"] = (
            left_whole_note_or_longer_events / left_note_events
        )
    if right_note_events > 0:
        metrics[f"{metric_prefix}/rate/right_long_note_events"] = right_long_note_events / right_note_events
        metrics[f"{metric_prefix}/rate/right_whole_bar_note_events"] = right_whole_bar_note_events / right_note_events
        metrics[f"{metric_prefix}/rate/right_whole_note_or_longer_events"] = (
            right_whole_note_or_longer_events / right_note_events
        )
    if left_long_note_events > 0:
        metrics[f"{metric_prefix}/rate/static_long_left_under_right_motion"] = (
            long_left_under_right_motion / left_long_note_events
        )

    return metrics


def _melodic_metrics(metric_prefix: str, counts: _MelodicCounts) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if counts.intervals > 0:
        metrics[f"{metric_prefix}/rate/stepwise_motion"] = counts.stepwise_intervals / counts.intervals
        metrics[f"{metric_prefix}/rate/repeated_motion"] = counts.repeated_intervals / counts.intervals
        metrics[f"{metric_prefix}/rate/large_leap"] = counts.large_leaps / counts.intervals
    if counts.recovery_opportunities > 0:
        metrics[f"{metric_prefix}/rate/large_leap_recovery"] = counts.recoveries / counts.recovery_opportunities
    if counts.direction_change_opportunities > 0:
        metrics[f"{metric_prefix}/rate/direction_change"] = (
            counts.direction_changes / counts.direction_change_opportunities
        )

    return metrics


def _hand_dialogue_metrics(metric_prefix: str, counts: _HandDialogueCounts) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if counts.onsets > 0:
        metrics[f"{metric_prefix}/rate/answered_onsets"] = counts.answered_onsets / counts.onsets
    if counts.onset_positions > 0:
        metrics[f"{metric_prefix}/rate/synchronized_onsets"] = counts.synchronized_positions / counts.onset_positions

    return metrics


def _closure_metrics(metric_prefix: str, counts: _ClosureCounts) -> dict[str, float]:
    if counts.samples == 0:
        return {}

    return {
        f"{metric_prefix}/rate/final_activity": counts.final_activity / counts.samples,
        f"{metric_prefix}/rate/final_both_hands_active": counts.both_hands_active / counts.samples,
        f"{metric_prefix}/rate/final_left_root_support": counts.left_root_support / counts.samples,
        f"{metric_prefix}/rate/final_right_tonic_closure": counts.right_tonic_closure / counts.samples,
        f"{metric_prefix}/rate/final_long_note": counts.long_final_note / counts.samples,
    }


def _add_melodic_counts(left: _MelodicCounts, right: _MelodicCounts) -> _MelodicCounts:
    return _MelodicCounts(
        intervals=left.intervals + right.intervals,
        stepwise_intervals=left.stepwise_intervals + right.stepwise_intervals,
        repeated_intervals=left.repeated_intervals + right.repeated_intervals,
        large_leaps=left.large_leaps + right.large_leaps,
        recoveries=left.recoveries + right.recoveries,
        recovery_opportunities=left.recovery_opportunities + right.recovery_opportunities,
        direction_changes=left.direction_changes + right.direction_changes,
        direction_change_opportunities=left.direction_change_opportunities + right.direction_change_opportunities,
    )


def _add_hand_dialogue_counts(left: _HandDialogueCounts, right: _HandDialogueCounts) -> _HandDialogueCounts:
    return _HandDialogueCounts(
        onsets=left.onsets + right.onsets,
        answered_onsets=left.answered_onsets + right.answered_onsets,
        onset_positions=left.onset_positions + right.onset_positions,
        synchronized_positions=left.synchronized_positions + right.synchronized_positions,
    )


def _add_closure_counts(left: _ClosureCounts, right: _ClosureCounts) -> _ClosureCounts:
    return _ClosureCounts(
        samples=left.samples + right.samples,
        final_activity=left.final_activity + right.final_activity,
        both_hands_active=left.both_hands_active + right.both_hands_active,
        left_root_support=left.left_root_support + right.left_root_support,
        right_tonic_closure=left.right_tonic_closure + right.right_tonic_closure,
        long_final_note=left.long_final_note + right.long_final_note,
    )
