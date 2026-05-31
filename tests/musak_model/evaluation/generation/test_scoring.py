from pytest import approx

from musak_model.evaluation.diagnostics import SegmentDiagnostics
from musak_model.evaluation.generation import generation_sample_score, generation_sample_score_metrics
from musak_model.evaluation.generation.schema import ConstraintReport, GenerationSample
from musak_model.tokens.schema import ScaleType
from musak_model.training.config import GenerationEvaluationConfig


def _generation_config() -> GenerationEvaluationConfig:
    return GenerationEvaluationConfig(
        enabled=True,
        every_epochs=5,
        soft_sample_count=1,
        hard_sample_count=0,
        max_new_tokens=16,
        temperature=1.0,
        top_k=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        bar_count=2,
        minimum_duration_denominator=16,
        allow_dotted_durations=False,
        max_notes_per_hand=5,
        maximum_onset_span_semitones=12,
        maximum_pitch_gap_semitones=12,
        maximum_static_hand_span_degrees=5,
    )


def _sample(*, diagnostics: SegmentDiagnostics | None) -> GenerationSample:
    return GenerationSample(
        tokens=[],
        reached_end=False,
        generated_token_count=0,
        constraint_error="no legal continuation",
        constraint_report=ConstraintReport(
            failed=True,
            valid_token_fraction=0.5,
            first_failure_step=2,
            error="invalid token",
        ),
        diagnostics=diagnostics,
        decode_error=None,
        completed_bars=1,
        target_bar_count=2,
    )


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
        silent_bar_fraction=0.25,
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
        shortest_note_duration_beats=0.25,
        has_dotted_notes=True,
        max_notes_per_onset=3,
        max_notes_per_hand=6,
        max_onset_span_semitones=18,
        max_melodic_gap_semitones=15,
        static_hand_span_degrees=10,
        synchronized_onset_fraction=0.4,
        independent_onset_fraction=0.6,
    )


def test_generation_sample_score_has_interpretable_penalty_terms() -> None:
    score = generation_sample_score(_sample(diagnostics=_diagnostics()), config=_generation_config())
    terms = {term.name: term.penalty for term in score.terms}

    assert terms["constraint_error"] == 1.0
    assert terms["constraint_failure"] == 1.0
    assert terms["incomplete"] == 1.0
    assert terms["bar_count_error"] == 0.5
    assert terms["invalid_token_fraction"] == 0.5
    assert terms["hand_activity_imbalance"] == 0.25
    assert terms["out_of_scale_note_fraction"] == approx(0.2)
    assert terms["dotted_duration_when_disallowed"] == 1.0
    assert terms["max_notes_per_hand_excess"] == 0.2
    assert terms["max_onset_span_excess"] == 0.5
    assert terms["max_melodic_gap_excess"] == 0.25
    assert terms["static_hand_span_excess"] == 1.0
    assert score.total_penalty == approx(sum(terms.values()))


def test_generation_sample_score_metrics_aggregate_total_and_terms() -> None:
    metrics = generation_sample_score_metrics(
        "soft",
        [_sample(diagnostics=_diagnostics()), _sample(diagnostics=None)],
        config=_generation_config(),
    )

    assert metrics["generation/soft/count/scored_samples"] == 2.0
    assert metrics["generation/soft/mean/sample_penalty_constraint_error"] == 1.0
    assert metrics["generation/soft/mean/sample_penalty_hand_activity_imbalance"] == 0.25
    assert metrics["generation/soft/mean/sample_penalty"] > 0.0


def test_generation_sample_score_metrics_handle_empty_suite() -> None:
    metrics = generation_sample_score_metrics("hard", [], config=_generation_config())

    assert metrics == {"generation/hard/count/scored_samples": 0.0}
