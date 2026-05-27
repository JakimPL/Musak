from musak_model.evaluation.diagnostics import SegmentDiagnostics
from musak_model.evaluation.generation import reference_free_generation_metrics


def test_reference_free_generation_metrics_return_curated_diagnostic_subset() -> None:
    metrics = reference_free_generation_metrics(_diagnostics())

    assert [metric.key for metric in metrics] == [
        "empty_score",
        "one_hand_only",
        "both_hands_active_fraction",
        "hand_activity_balance",
        "silent_bar_fraction",
        "in_scale_note_fraction",
        "note_density_per_beat",
        "onset_density_per_beat",
        "shortest_note_duration_beats",
        "has_dotted_notes",
        "max_notes_per_onset",
        "max_onset_span_semitones",
        "max_melodic_gap_semitones",
        "synchronized_onset_fraction",
    ]


def _diagnostics() -> SegmentDiagnostics:
    return SegmentDiagnostics(
        right_silence_fraction=0.25,
        left_silence_fraction=0.5,
        both_hands_silence_fraction=0.1,
        both_hands_active_fraction=0.25,
        right_only_active_fraction=0.3,
        left_only_active_fraction=0.35,
        longest_right_silence_beats=1.0,
        longest_left_silence_beats=2.0,
        longest_both_hands_silence_beats=0.5,
        right_note_onsets_per_bar=2.0,
        left_note_onsets_per_bar=1.0,
        silent_bar_count=1,
        silent_bar_fraction=0.125,
        silent_edge_bar_count=0,
        hand_activity_balance=0.75,
        empty_score=False,
        one_hand_only=False,
        note_token_fraction=0.6,
        rest_token_fraction=0.3,
        hold_token_fraction=0.1,
        accidental_note_fraction=0.2,
        in_scale_note_fraction=0.8,
        note_density_per_beat=1.5,
        onset_density_per_beat=1.25,
        right_onset_density_per_beat=0.75,
        left_onset_density_per_beat=0.5,
        shortest_note_duration_beats=0.5,
        has_dotted_notes=True,
        max_notes_per_onset=3,
        max_notes_per_hand=2,
        max_onset_span_semitones=7,
        max_melodic_gap_semitones=5,
        static_hand_span_degrees=6,
        synchronized_onset_fraction=0.4,
        independent_onset_fraction=0.6,
    )
