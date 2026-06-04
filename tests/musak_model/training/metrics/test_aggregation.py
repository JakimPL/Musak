from pytest import approx

from musak_model.conditioning.harmony.relations import HARMONIC_RELATION_CLASS_COUNT
from musak_model.training.metrics import BatchMetrics, MetricsAccumulator


def test_metrics_accumulator_averages_loss_accuracy_and_gradient_norms() -> None:
    accumulator = MetricsAccumulator()

    accumulator.add(
        BatchMetrics(
            loss=1.0,
            token_count=2,
            exact_match_count=1,
            token_kind_match_count=2,
            event_kind_loss=0.5,
            event_kind_loss_target_count=2,
            duration_loss=0.25,
            duration_loss_target_count=2,
            duration_match_count=1,
            duration_target_count=2,
            degree_match_count=1,
            degree_target_count=1,
            accidental_match_count=1,
            accidental_target_count=1,
            octave_offset_match_count=0,
            octave_offset_target_count=1,
            hand_match_count=0,
            hand_target_count=0,
            musical_auxiliary_loss=1.0,
            musical_auxiliary_target_count=6,
            note_density_loss=0.2,
            note_density_match_count=1,
            note_density_target_count=2,
            rhythmic_diversity_loss=0.4,
            rhythmic_diversity_match_count=2,
            rhythmic_diversity_target_count=2,
            voice_independence_loss=0.6,
            voice_independence_match_count=1,
            voice_independence_target_count=2,
            uses_accidentals_loss=0.8,
            uses_accidentals_match_count=2,
            uses_accidentals_target_count=2,
            dotted_duration_loss=1.0,
            dotted_duration_match_count=0,
            dotted_duration_target_count=2,
            hand_span_loss=1.2,
            hand_span_match_count=1,
            hand_span_target_count=2,
            bar_note_density_loss=0.3,
            bar_note_density_match_count=1,
            bar_note_density_target_count=2,
            bar_rhythmic_diversity_loss=0.4,
            bar_rhythmic_diversity_match_count=2,
            bar_rhythmic_diversity_target_count=2,
            bar_voice_independence_loss=0.6,
            bar_voice_independence_match_count=1,
            bar_voice_independence_target_count=2,
            bar_uses_accidentals_loss=0.8,
            bar_uses_accidentals_match_count=2,
            bar_uses_accidentals_target_count=2,
            bar_dotted_duration_loss=1.0,
            bar_dotted_duration_match_count=0,
            bar_dotted_duration_target_count=2,
            bar_hand_span_loss=1.2,
            bar_hand_span_match_count=1,
            bar_hand_span_target_count=2,
            harmonic_relation_loss=0.9,
            harmonic_relation_match_count=1,
            harmonic_relation_target_count=2,
            harmonic_relation_macro_f1=0.5,
            harmonic_relation_target_counts=(1, 1, 0, 0, 0, 0, 0),
            harmonic_relation_prediction_counts=(1, 0, 1, 0, 0, 0, 0),
            harmonic_plan_reconstruction_loss=1.0,
            harmonic_plan_reconstruction_target_count=10,
            harmonic_plan_reconstruction_harmonic_function_match_count=1,
            harmonic_plan_reconstruction_harmonic_function_target_count=2,
            harmonic_plan_reconstruction_root_degree_match_count=1,
            harmonic_plan_reconstruction_root_degree_target_count=2,
            harmonic_plan_reconstruction_quality_match_count=1,
            harmonic_plan_reconstruction_quality_target_count=2,
            harmonic_plan_reconstruction_extension_match_count=1,
            harmonic_plan_reconstruction_extension_target_count=2,
            harmonic_plan_reconstruction_cadence_strength_match_count=1,
            harmonic_plan_reconstruction_cadence_strength_target_count=2,
            harmonic_plan_contrastive_loss=0.4,
            harmonic_plan_contrastive_match_count=1,
            harmonic_plan_contrastive_target_count=2,
            harmonic_plan_contrastive_positive_similarity=0.1,
            harmonic_plan_contrastive_negative_similarity=0.3,
            harmony_gate_mean=0.25,
            harmony_gate_token_count=2,
            validity_penalty_loss=0.2,
            invalid_probability_mass=0.4,
            invalid_target_count=1,
            validity_penalty_token_count=2,
            cnn_gradient_norm=0.5,
            gru_gradient_norm=1.0,
            transformer_gradient_norm=1.5,
        )
    )
    accumulator.add(
        BatchMetrics(
            loss=2.0,
            token_count=6,
            exact_match_count=3,
            token_kind_match_count=4,
            event_kind_loss=1.0,
            event_kind_loss_target_count=6,
            duration_loss=0.75,
            duration_loss_target_count=4,
            duration_match_count=3,
            duration_target_count=4,
            degree_match_count=2,
            degree_target_count=3,
            accidental_match_count=2,
            accidental_target_count=3,
            octave_offset_match_count=2,
            octave_offset_target_count=3,
            hand_match_count=1,
            hand_target_count=1,
            musical_auxiliary_loss=2.0,
            musical_auxiliary_target_count=12,
            note_density_loss=0.4,
            note_density_match_count=2,
            note_density_target_count=4,
            rhythmic_diversity_loss=0.8,
            rhythmic_diversity_match_count=1,
            rhythmic_diversity_target_count=4,
            voice_independence_loss=1.2,
            voice_independence_match_count=3,
            voice_independence_target_count=4,
            uses_accidentals_loss=1.6,
            uses_accidentals_match_count=4,
            uses_accidentals_target_count=4,
            dotted_duration_loss=2.0,
            dotted_duration_match_count=2,
            dotted_duration_target_count=4,
            hand_span_loss=2.4,
            hand_span_match_count=3,
            hand_span_target_count=4,
            bar_note_density_loss=0.6,
            bar_note_density_match_count=3,
            bar_note_density_target_count=4,
            bar_rhythmic_diversity_loss=0.8,
            bar_rhythmic_diversity_match_count=1,
            bar_rhythmic_diversity_target_count=4,
            bar_voice_independence_loss=1.2,
            bar_voice_independence_match_count=3,
            bar_voice_independence_target_count=4,
            bar_uses_accidentals_loss=1.6,
            bar_uses_accidentals_match_count=4,
            bar_uses_accidentals_target_count=4,
            bar_dotted_duration_loss=2.0,
            bar_dotted_duration_match_count=2,
            bar_dotted_duration_target_count=4,
            bar_hand_span_loss=2.4,
            bar_hand_span_match_count=3,
            bar_hand_span_target_count=4,
            harmonic_relation_loss=1.5,
            harmonic_relation_match_count=3,
            harmonic_relation_target_count=4,
            harmonic_relation_macro_f1=0.75,
            harmonic_relation_target_counts=(1, 0, 3, 0, 0, 0, 0),
            harmonic_relation_prediction_counts=(0, 2, 2, 0, 0, 0, 0),
            harmonic_plan_reconstruction_loss=2.0,
            harmonic_plan_reconstruction_target_count=20,
            harmonic_plan_reconstruction_harmonic_function_match_count=3,
            harmonic_plan_reconstruction_harmonic_function_target_count=4,
            harmonic_plan_reconstruction_root_degree_match_count=3,
            harmonic_plan_reconstruction_root_degree_target_count=4,
            harmonic_plan_reconstruction_quality_match_count=3,
            harmonic_plan_reconstruction_quality_target_count=4,
            harmonic_plan_reconstruction_extension_match_count=3,
            harmonic_plan_reconstruction_extension_target_count=4,
            harmonic_plan_reconstruction_cadence_strength_match_count=3,
            harmonic_plan_reconstruction_cadence_strength_target_count=4,
            harmonic_plan_contrastive_loss=0.8,
            harmonic_plan_contrastive_match_count=3,
            harmonic_plan_contrastive_target_count=4,
            harmonic_plan_contrastive_positive_similarity=0.5,
            harmonic_plan_contrastive_negative_similarity=0.7,
            harmony_gate_mean=0.75,
            harmony_gate_token_count=6,
            validity_penalty_loss=0.6,
            invalid_probability_mass=0.8,
            invalid_target_count=2,
            validity_penalty_token_count=6,
            cnn_gradient_norm=1.5,
            gru_gradient_norm=2.0,
            transformer_gradient_norm=2.5,
        )
    )

    metrics = accumulator.to_epoch_split_metrics()

    assert metrics.loss == 1.75
    assert metrics.token_accuracy == 0.5
    assert metrics.token_kind_accuracy == 0.75
    assert metrics.event_kind_loss == approx(0.875)
    assert metrics.duration_loss == approx(0.5833333333333334)
    assert metrics.duration_accuracy == approx(4 / 6)
    assert metrics.degree_accuracy == approx(3 / 4)
    assert metrics.accidental_accuracy == approx(3 / 4)
    assert metrics.octave_offset_accuracy == approx(2 / 4)
    assert metrics.hand_accuracy == 1.0
    assert metrics.musical_auxiliary_loss == approx(5 / 3)
    assert metrics.note_density_loss == approx(1 / 3)
    assert metrics.note_density_accuracy == 0.5
    assert metrics.rhythmic_diversity_loss == approx(2 / 3)
    assert metrics.rhythmic_diversity_accuracy == 0.5
    assert metrics.voice_independence_loss == 1.0
    assert metrics.voice_independence_accuracy == approx(4 / 6)
    assert metrics.uses_accidentals_loss == approx(4 / 3)
    assert metrics.uses_accidentals_accuracy == 1.0
    assert metrics.dotted_duration_loss == approx(5 / 3)
    assert metrics.dotted_duration_accuracy == approx(2 / 6)
    assert metrics.hand_span_loss == 2.0
    assert metrics.hand_span_accuracy == approx(4 / 6)
    assert metrics.bar_note_density_loss == 0.5
    assert metrics.bar_note_density_accuracy == approx(4 / 6)
    assert metrics.bar_rhythmic_diversity_loss == approx(2 / 3)
    assert metrics.bar_rhythmic_diversity_accuracy == 0.5
    assert metrics.bar_voice_independence_loss == 1.0
    assert metrics.bar_voice_independence_accuracy == approx(4 / 6)
    assert metrics.bar_uses_accidentals_loss == approx(4 / 3)
    assert metrics.bar_uses_accidentals_accuracy == 1.0
    assert metrics.bar_dotted_duration_loss == approx(5 / 3)
    assert metrics.bar_dotted_duration_accuracy == approx(2 / 6)
    assert metrics.bar_hand_span_loss == 2.0
    assert metrics.bar_hand_span_accuracy == approx(4 / 6)
    assert metrics.harmonic_relation_loss == approx(1.3)
    assert metrics.harmonic_relation_accuracy == approx(4 / 6)
    assert metrics.harmonic_relation_macro_f1 == approx(2 / 3)
    assert metrics.harmonic_relation_target_distribution == approx((2 / 6, 1 / 6, 3 / 6, 0, 0, 0, 0))
    assert metrics.harmonic_relation_prediction_distribution == approx((1 / 6, 2 / 6, 3 / 6, 0, 0, 0, 0))
    assert len(metrics.harmonic_relation_target_distribution or ()) == HARMONIC_RELATION_CLASS_COUNT
    assert metrics.harmonic_plan_reconstruction_loss == approx(5 / 3)
    assert metrics.harmonic_plan_reconstruction_harmonic_function_accuracy == approx(4 / 6)
    assert metrics.harmonic_plan_reconstruction_root_degree_accuracy == approx(4 / 6)
    assert metrics.harmonic_plan_reconstruction_quality_accuracy == approx(4 / 6)
    assert metrics.harmonic_plan_reconstruction_extension_accuracy == approx(4 / 6)
    assert metrics.harmonic_plan_reconstruction_cadence_strength_accuracy == approx(4 / 6)
    assert metrics.harmonic_plan_contrastive_loss == approx(2 / 3)
    assert metrics.harmonic_plan_contrastive_accuracy == approx(4 / 6)
    assert metrics.harmonic_plan_contrastive_positive_similarity == approx(2.2 / 6)
    assert metrics.harmonic_plan_contrastive_negative_similarity == approx(3.4 / 6)
    assert metrics.harmony_gate_mean == approx(0.625)
    assert metrics.validity_penalty_loss == approx(0.5)
    assert metrics.invalid_probability_mass == approx(0.7)
    assert metrics.invalid_target_rate == 0.375
    assert metrics.cnn_gradient_norm == 1.25
    assert metrics.gru_gradient_norm == 1.75
    assert metrics.transformer_gradient_norm == 2.25
