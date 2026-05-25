from __future__ import annotations

from musak_model.conditioning.structural import extract_structural_control_features
from musak_model.data.schema import Segment
from musak_model.evaluation.diagnostics.activity import (
    calculate_activity_balance,
    calculate_beats,
    calculate_count_per_beat,
    calculate_duration_beats,
    calculate_duration_fraction,
    calculate_hand_state_durations,
    calculate_longest_silence,
    calculate_onset_fraction,
    calculate_onsets_per_bar,
    collect_onset_starts,
    count_silent_edge_bars,
    find_silent_bar_indices,
    merge_intervals,
)
from musak_model.evaluation.diagnostics.constants import HANDS
from musak_model.evaluation.diagnostics.events import calculate_segment_duration, collect_segment_activity_events
from musak_model.evaluation.diagnostics.schema import SegmentDiagnostics
from musak_model.evaluation.diagnostics.tokens import (
    calculate_note_fraction,
    calculate_token_fraction,
    coalesce_optional_integer,
    collect_note_tokens,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HoldToken, NoteToken, RestToken


def diagnose_segment(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> SegmentDiagnostics:
    events = collect_segment_activity_events(segment, duration_vocabulary=duration_vocabulary)
    total_duration = calculate_segment_duration(segment, events)
    structural_features = extract_structural_control_features(segment, duration_vocabulary=duration_vocabulary)
    segment_note_tokens = collect_note_tokens(segment)
    total_beats = calculate_beats(total_duration, denominator=segment.time_denominator)
    starts_by_hand = collect_onset_starts(events)
    synchronized_onsets = starts_by_hand[Hand.RIGHT] & starts_by_hand[Hand.LEFT]
    independent_onsets = starts_by_hand[Hand.RIGHT] ^ starts_by_hand[Hand.LEFT]
    all_onsets = starts_by_hand[Hand.RIGHT] | starts_by_hand[Hand.LEFT]
    active_intervals = {
        hand: merge_intervals((event.start, event.end) for event in events if event.hand == hand) for hand in HANDS
    }
    silent_bars = find_silent_bar_indices(segment, events)
    state_durations = calculate_hand_state_durations(active_intervals, total_duration=total_duration)
    right_active_duration = state_durations.right_only_active_duration + state_durations.both_hands_active_duration
    left_active_duration = state_durations.left_only_active_duration + state_durations.both_hands_active_duration
    bar_count = max(segment.bar_count, 1)

    return SegmentDiagnostics(
        right_silence_fraction=calculate_duration_fraction(total_duration - right_active_duration, total_duration),
        left_silence_fraction=calculate_duration_fraction(total_duration - left_active_duration, total_duration),
        both_hands_silence_fraction=calculate_duration_fraction(
            state_durations.both_hands_silence_duration,
            total_duration,
        ),
        both_hands_active_fraction=calculate_duration_fraction(
            state_durations.both_hands_active_duration,
            total_duration,
        ),
        right_only_active_fraction=calculate_duration_fraction(
            state_durations.right_only_active_duration,
            total_duration,
        ),
        left_only_active_fraction=calculate_duration_fraction(
            state_durations.left_only_active_duration,
            total_duration,
        ),
        longest_right_silence_beats=calculate_beats(
            calculate_longest_silence(active_intervals[Hand.RIGHT], total_duration=total_duration),
            denominator=segment.time_denominator,
        ),
        longest_left_silence_beats=calculate_beats(
            calculate_longest_silence(active_intervals[Hand.LEFT], total_duration=total_duration),
            denominator=segment.time_denominator,
        ),
        longest_both_hands_silence_beats=calculate_beats(
            state_durations.longest_both_hands_silence,
            denominator=segment.time_denominator,
        ),
        right_note_onsets_per_bar=calculate_onsets_per_bar(events, hand=Hand.RIGHT, bar_count=bar_count),
        left_note_onsets_per_bar=calculate_onsets_per_bar(events, hand=Hand.LEFT, bar_count=bar_count),
        silent_bar_count=len(silent_bars),
        silent_bar_fraction=len(silent_bars) / bar_count,
        silent_edge_bar_count=count_silent_edge_bars(silent_bars, bar_count=bar_count),
        hand_activity_balance=calculate_activity_balance(right_active_duration, left_active_duration),
        empty_score=right_active_duration == 0 and left_active_duration == 0,
        one_hand_only=(right_active_duration == 0) != (left_active_duration == 0),
        note_token_fraction=calculate_token_fraction(segment, NoteToken),
        rest_token_fraction=calculate_token_fraction(segment, RestToken),
        hold_token_fraction=calculate_token_fraction(segment, HoldToken),
        accidental_note_fraction=calculate_note_fraction(segment_note_tokens, lambda note: note.accidental != 0),
        in_scale_note_fraction=calculate_note_fraction(segment_note_tokens, lambda note: note.accidental == 0),
        note_density_per_beat=calculate_count_per_beat(len(events), total_beats=total_beats),
        onset_density_per_beat=calculate_count_per_beat(
            sum(len(hand_onsets) for hand_onsets in starts_by_hand.values()),
            total_beats=total_beats,
        ),
        right_onset_density_per_beat=calculate_count_per_beat(
            len(starts_by_hand[Hand.RIGHT]),
            total_beats=total_beats,
        ),
        left_onset_density_per_beat=calculate_count_per_beat(
            len(starts_by_hand[Hand.LEFT]),
            total_beats=total_beats,
        ),
        shortest_note_duration_beats=calculate_duration_beats(
            structural_features.shortest_note_duration,
            denominator=segment.time_denominator,
        ),
        has_dotted_notes=bool(structural_features.has_dotted_notes),
        max_notes_per_onset=coalesce_optional_integer(structural_features.max_notes_per_onset),
        max_notes_per_hand=coalesce_optional_integer(structural_features.max_notes_per_hand),
        max_onset_span_semitones=coalesce_optional_integer(structural_features.max_onset_span_semitones),
        max_melodic_gap_semitones=coalesce_optional_integer(structural_features.max_melodic_gap_semitones),
        static_hand_span_degrees=coalesce_optional_integer(structural_features.static_hand_span_degrees),
        synchronized_onset_fraction=calculate_onset_fraction(synchronized_onsets, all_onsets=all_onsets),
        independent_onset_fraction=calculate_onset_fraction(independent_onsets, all_onsets=all_onsets),
    )
