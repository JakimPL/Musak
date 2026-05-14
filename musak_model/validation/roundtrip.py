from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from fractions import Fraction
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from musak_model.data.schema import ParsedScore
from musak_model.decoder.piano_roll import PianoRollEvent, parsed_score_to_piano_roll_events
from musak_model.tokens.schema import Hand

_ExactEventKey: TypeAlias = tuple[Hand, Fraction, int, Fraction]
_OnsetPitchKey: TypeAlias = tuple[Hand, Fraction, int]
_OnsetKey: TypeAlias = tuple[Hand, Fraction]


class RoundtripMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reference_note_count: int = Field(ge=0)
    decoded_note_count: int = Field(ge=0)
    exact_match_count: int = Field(ge=0)
    exact_precision: float = Field(ge=0, le=1)
    exact_recall: float = Field(ge=0, le=1)
    exact_f1: float = Field(ge=0, le=1)
    onset_pitch_match_count: int = Field(ge=0)
    onset_pitch_precision: float = Field(ge=0, le=1)
    onset_pitch_recall: float = Field(ge=0, le=1)
    onset_pitch_f1: float = Field(ge=0, le=1)
    onset_only_match_count: int = Field(ge=0)
    pitch_accuracy_for_onset_matches: float = Field(ge=0, le=1)
    mean_onset_error: Fraction | None = None
    max_onset_error: Fraction | None = None
    mean_duration_error: Fraction | None = None
    max_duration_error: Fraction | None = None
    reference_duplicate_pitch_onsets: int = Field(ge=0)
    decoded_duplicate_pitch_onsets: int = Field(ge=0)
    bar_count_matches: bool
    reference_valid_bar_count: int = Field(ge=0)
    decoded_valid_bar_count: int = Field(ge=0)
    time_signature_matches: bool | None = None


def compare_parsed_scores(reference: ParsedScore, decoded: ParsedScore) -> RoundtripMetrics:
    return _compare_events(
        reference_events=parsed_score_to_piano_roll_events(reference),
        decoded_events=parsed_score_to_piano_roll_events(decoded),
        reference_bar_count=_bar_count(reference),
        decoded_bar_count=_bar_count(decoded),
        reference_measure_duration=Fraction(reference.time_numerator, reference.time_denominator),
        decoded_measure_duration=Fraction(decoded.time_numerator, decoded.time_denominator),
        time_signature_matches=(
            reference.time_numerator,
            reference.time_denominator,
        )
        == (
            decoded.time_numerator,
            decoded.time_denominator,
        ),
    )


def compare_parsed_score_to_events(
    reference: ParsedScore,
    decoded_events: list[PianoRollEvent],
) -> RoundtripMetrics:
    measure_duration = Fraction(reference.time_numerator, reference.time_denominator)
    return _compare_events(
        reference_events=parsed_score_to_piano_roll_events(reference),
        decoded_events=decoded_events,
        reference_bar_count=_bar_count(reference),
        decoded_bar_count=_bar_count(reference),
        reference_measure_duration=measure_duration,
        decoded_measure_duration=measure_duration,
        time_signature_matches=None,
    )


def _compare_events(
    *,
    reference_events: list[PianoRollEvent],
    decoded_events: list[PianoRollEvent],
    reference_bar_count: int,
    decoded_bar_count: int,
    reference_measure_duration: Fraction,
    decoded_measure_duration: Fraction,
    time_signature_matches: bool | None,
) -> RoundtripMetrics:
    exact_match_count = _multiset_intersection_count(
        Counter(_exact_key(event) for event in reference_events),
        Counter(_exact_key(event) for event in decoded_events),
    )
    onset_pitch_match_count = _multiset_intersection_count(
        Counter(_onset_pitch_key(event) for event in reference_events),
        Counter(_onset_pitch_key(event) for event in decoded_events),
    )
    onset_only_match_count = _multiset_intersection_count(
        Counter(_onset_key(event) for event in reference_events),
        Counter(_onset_key(event) for event in decoded_events),
    )
    onset_errors, duration_errors = _nearest_event_errors(reference_events, decoded_events)
    exact_precision, exact_recall, exact_f1 = _precision_recall_f1(
        match_count=exact_match_count,
        decoded_count=len(decoded_events),
        reference_count=len(reference_events),
    )
    onset_pitch_precision, onset_pitch_recall, onset_pitch_f1 = _precision_recall_f1(
        match_count=onset_pitch_match_count,
        decoded_count=len(decoded_events),
        reference_count=len(reference_events),
    )

    return RoundtripMetrics(
        reference_note_count=len(reference_events),
        decoded_note_count=len(decoded_events),
        exact_match_count=exact_match_count,
        exact_precision=exact_precision,
        exact_recall=exact_recall,
        exact_f1=exact_f1,
        onset_pitch_match_count=onset_pitch_match_count,
        onset_pitch_precision=onset_pitch_precision,
        onset_pitch_recall=onset_pitch_recall,
        onset_pitch_f1=onset_pitch_f1,
        onset_only_match_count=onset_only_match_count,
        pitch_accuracy_for_onset_matches=_safe_ratio(onset_pitch_match_count, onset_only_match_count),
        mean_onset_error=_mean(onset_errors),
        max_onset_error=max(onset_errors) if onset_errors else None,
        mean_duration_error=_mean(duration_errors),
        max_duration_error=max(duration_errors) if duration_errors else None,
        reference_duplicate_pitch_onsets=_duplicate_pitch_onset_count(reference_events),
        decoded_duplicate_pitch_onsets=_duplicate_pitch_onset_count(decoded_events),
        bar_count_matches=reference_bar_count == decoded_bar_count,
        reference_valid_bar_count=_valid_bar_count(
            reference_events,
            bar_count=reference_bar_count,
            measure_duration=reference_measure_duration,
        ),
        decoded_valid_bar_count=_valid_bar_count(
            decoded_events,
            bar_count=decoded_bar_count,
            measure_duration=decoded_measure_duration,
        ),
        time_signature_matches=time_signature_matches,
    )


def _bar_count(score: ParsedScore) -> int:
    return max(len(score.right_hand_bars), len(score.left_hand_bars))


def _exact_key(event: PianoRollEvent) -> _ExactEventKey:
    return event.hand, event.start, event.midi_pitch, event.duration


def _onset_pitch_key(event: PianoRollEvent) -> _OnsetPitchKey:
    return event.hand, event.start, event.midi_pitch


def _onset_key(event: PianoRollEvent) -> _OnsetKey:
    return event.hand, event.start


def _multiset_intersection_count(left: Counter[Hashable], right: Counter[Hashable]) -> int:
    return sum((left & right).values())


def _precision_recall_f1(
    *,
    match_count: int,
    decoded_count: int,
    reference_count: int,
) -> tuple[float, float, float]:
    precision = _safe_ratio(match_count, decoded_count)
    recall = _safe_ratio(match_count, reference_count)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return numerator / denominator


def _nearest_event_errors(
    reference_events: list[PianoRollEvent],
    decoded_events: list[PianoRollEvent],
) -> tuple[list[Fraction], list[Fraction]]:
    remaining_reference = list(reference_events)
    onset_errors: list[Fraction] = []
    duration_errors: list[Fraction] = []
    for decoded_event in decoded_events:
        candidates = [
            reference_event
            for reference_event in remaining_reference
            if reference_event.hand == decoded_event.hand and reference_event.midi_pitch == decoded_event.midi_pitch
        ]
        if not candidates:
            continue

        nearest = _nearest_by_onset(candidates, decoded_event)
        remaining_reference.remove(nearest)
        onset_errors.append(abs(nearest.start - decoded_event.start))
        duration_errors.append(abs(nearest.duration - decoded_event.duration))

    return onset_errors, duration_errors


def _nearest_by_onset(
    candidates: list[PianoRollEvent],
    decoded_event: PianoRollEvent,
) -> PianoRollEvent:
    return min(candidates, key=lambda reference_event: abs(reference_event.start - decoded_event.start))


def _mean(values: list[Fraction]) -> Fraction | None:
    if not values:
        return None

    return sum(values, Fraction(0)) / len(values)


def _duplicate_pitch_onset_count(events: list[PianoRollEvent]) -> int:
    counts = Counter(_onset_pitch_key(event) for event in events)
    return sum(count - 1 for count in counts.values() if count > 1)


def _valid_bar_count(
    events: list[PianoRollEvent],
    *,
    bar_count: int,
    measure_duration: Fraction,
) -> int:
    valid_bars = [True] * bar_count
    for event in events:
        if event.start < 0 or event.end < event.start:
            return sum(valid_bars)

        bar_index = event.start // measure_duration
        if not isinstance(bar_index, int):
            bar_index = int(bar_index)

        if not 0 <= bar_index < bar_count:
            continue

        bar_end = (bar_index + 1) * measure_duration
        if event.end > bar_end:
            valid_bars[bar_index] = False

    return sum(valid_bars)
