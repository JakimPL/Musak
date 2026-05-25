from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

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


@dataclass(frozen=True)
class ActivityEvent:
    hand: Hand
    start: Fraction
    duration: Fraction

    @property
    def end(self) -> Fraction:
        return self.start + self.duration


def collect_segment_activity_events(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> list[ActivityEvent]:
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    active_hand = Hand.RIGHT
    bar_index = 0
    cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}
    last_attack_indices: dict[Hand, list[int]] = {Hand.RIGHT: [], Hand.LEFT: []}
    events: list[ActivityEvent] = []

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
                extend_last_attack(events, event_indices=last_attack_indices[active_hand], duration=duration)
                cursors[active_hand] += duration
            case NoteToken(duration_id=duration_id):
                duration = duration_vocabulary.id_to_fraction(duration_id)
                events.append(
                    ActivityEvent(
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
                events[-1] = ActivityEvent(
                    hand=previous_event.hand,
                    start=joined_start,
                    duration=previous_event.duration,
                )
                last_attack_indices[active_hand] = find_same_onset_event_indices(
                    events,
                    hand=active_hand,
                    start=joined_start,
                )
                cursors[active_hand] = max(
                    cursors[active_hand] - previous_event.duration,
                    events[-1].end - bar_index * measure_duration,
                )

    return events


def calculate_segment_duration(segment: Segment, events: list[ActivityEvent]) -> Fraction:
    metadata_duration = segment.bar_count * Fraction(segment.time_numerator, segment.time_denominator)
    if metadata_duration > 0:
        return metadata_duration

    event_end = max((event.end for event in events), default=Fraction(0))
    return event_end


def extend_last_attack(events: list[ActivityEvent], *, event_indices: list[int], duration: Fraction) -> None:
    if not event_indices:
        raise ValueError("hold token needs a previous same-hand note or chord")

    for event_index in event_indices:
        event = events[event_index]
        events[event_index] = ActivityEvent(
            hand=event.hand,
            start=event.start,
            duration=event.duration + duration,
        )


def find_same_onset_event_indices(events: list[ActivityEvent], *, hand: Hand, start: Fraction) -> list[int]:
    return [index for index, event in enumerate(events) if event.hand == hand and event.start == start]
