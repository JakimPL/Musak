from dataclasses import dataclass
from fractions import Fraction

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
    Token,
)


@dataclass(frozen=True)
class PitchedOnset:
    notes: tuple[NoteToken, ...]
    start: Fraction
    duration: Fraction


@dataclass(frozen=True)
class HandOnsetRun:
    hand: Hand
    onsets: tuple[PitchedOnset, ...]


@dataclass
class _ParsedNoteEvent:
    hand: Hand
    note: NoteToken
    start: Fraction
    duration: Fraction

    @property
    def end(self) -> Fraction:
        return self.start + self.duration


def extract_hand_onset_runs(
    tokens: list[Token],
    *,
    duration_vocabulary: DurationVocabulary,
    time_numerator: int,
    time_denominator: int,
    default_hand: Hand = Hand.RIGHT,
) -> dict[Hand, tuple[HandOnsetRun, ...]]:
    events: list[_ParsedNoteEvent] = []
    boundary_times: dict[Hand, list[Fraction]] = {Hand.RIGHT: [], Hand.LEFT: []}
    last_attack_indices: dict[Hand, list[int]] = {Hand.RIGHT: [], Hand.LEFT: []}
    measure_duration = Fraction(time_numerator, time_denominator)
    active_hand = default_hand
    bar_index = 0
    cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}

    for token in tokens:
        match token:
            case HandToken():
                active_hand = token.hand
            case StartToken():
                continue
            case EndToken():
                break
            case BarToken():
                bar_index += 1
                boundary_time = bar_index * measure_duration
                for hand in Hand:
                    boundary_times[hand].append(boundary_time)
                    cursors[hand] = boundary_time
                    last_attack_indices[hand] = []
            case RestToken():
                boundary_times[active_hand].append(cursors[active_hand])
                cursors[active_hand] += duration_vocabulary.id_to_fraction(token.duration_id)
                last_attack_indices[active_hand] = []
            case HoldToken():
                _extend_last_attack(
                    events,
                    event_indices=last_attack_indices[active_hand],
                    hand=active_hand,
                    duration=duration_vocabulary.id_to_fraction(token.duration_id),
                )
                cursors[active_hand] = max(events[index].end for index in last_attack_indices[active_hand])
            case NoteToken():
                duration = duration_vocabulary.id_to_fraction(token.duration_id)
                events.append(
                    _ParsedNoteEvent(
                        hand=active_hand,
                        note=token,
                        start=cursors[active_hand],
                        duration=duration,
                    )
                )
                cursors[active_hand] += duration
                last_attack_indices[active_hand] = [len(events) - 1]
            case JoinWithPreviousToken():
                _join_latest_note_with_previous_onset(
                    events,
                    hand=active_hand,
                    last_attack_indices=last_attack_indices,
                )
                cursors[active_hand] = max(events[index].end for index in last_attack_indices[active_hand])

    return {
        hand: _hand_onset_runs(
            events,
            hand=hand,
            boundary_times=tuple(boundary_times[hand]),
        )
        for hand in Hand
    }


def _extend_last_attack(
    events: list[_ParsedNoteEvent],
    *,
    event_indices: list[int],
    hand: Hand,
    duration: Fraction,
) -> None:
    if not event_indices:
        raise ValueError(f"hold token has no previous {hand.value} hand attack")

    for index in event_indices:
        event = events[index]
        if event.hand != hand:
            raise ValueError("hold token cannot extend an attack in another hand")

        event.duration += duration


def _join_latest_note_with_previous_onset(
    events: list[_ParsedNoteEvent],
    *,
    hand: Hand,
    last_attack_indices: dict[Hand, list[int]],
) -> None:
    if len(events) < 2:
        raise ValueError("join-with-previous token needs at least two decoded notes")

    latest_event = events[-1]
    if latest_event.hand != hand:
        raise ValueError("join-with-previous token does not match the active hand")

    latest_event.start = events[-2].start
    last_attack_indices[hand] = [
        index for index, event in enumerate(events) if event.hand == hand and event.start == latest_event.start
    ]


def _hand_onset_runs(
    events: list[_ParsedNoteEvent],
    *,
    hand: Hand,
    boundary_times: tuple[Fraction, ...],
) -> tuple[HandOnsetRun, ...]:
    onsets = _group_hand_onsets(events, hand=hand)
    runs: list[HandOnsetRun] = []
    current_run: list[PitchedOnset] = []

    for onset in onsets:
        if current_run and _has_boundary_between(
            boundary_times,
            previous_start=current_run[-1].start,
            current_start=onset.start,
        ):
            runs.append(HandOnsetRun(hand=hand, onsets=tuple(current_run)))
            current_run = []

        current_run.append(onset)

    if current_run:
        runs.append(HandOnsetRun(hand=hand, onsets=tuple(current_run)))

    return tuple(runs)


def _group_hand_onsets(
    events: list[_ParsedNoteEvent],
    *,
    hand: Hand,
) -> tuple[PitchedOnset, ...]:
    hand_events = sorted(
        (event for event in events if event.hand == hand),
        key=lambda event: event.start,
    )
    onsets: list[PitchedOnset] = []
    current_start: Fraction | None = None
    current_events: list[_ParsedNoteEvent] = []

    for event in hand_events:
        if current_start is not None and event.start != current_start:
            onsets.append(_events_to_onset(current_events))
            current_events = []

        current_start = event.start
        current_events.append(event)

    if current_events:
        onsets.append(_events_to_onset(current_events))

    return tuple(onsets)


def _events_to_onset(events: list[_ParsedNoteEvent]) -> PitchedOnset:
    start = events[0].start
    return PitchedOnset(
        notes=tuple(event.note for event in events),
        start=start,
        duration=max(event.end for event in events) - start,
    )


def _has_boundary_between(
    boundary_times: tuple[Fraction, ...],
    *,
    previous_start: Fraction,
    current_start: Fraction,
) -> bool:
    return any(previous_start < boundary_time <= current_start for boundary_time in boundary_times)
