from pytest import approx

from musak_model.training.metrics import BatchMetrics, MetricsAccumulator


def test_metrics_accumulator_averages_loss_accuracy_and_gradient_norms() -> None:
    accumulator = MetricsAccumulator()

    accumulator.add(
        BatchMetrics(
            loss=1.0,
            token_count=2,
            exact_match_count=1,
            token_kind_match_count=2,
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
    assert metrics.duration_accuracy == approx(4 / 6)
    assert metrics.degree_accuracy == approx(3 / 4)
    assert metrics.accidental_accuracy == approx(3 / 4)
    assert metrics.octave_offset_accuracy == approx(2 / 4)
    assert metrics.hand_accuracy == 1.0
    assert metrics.validity_penalty_loss == approx(0.5)
    assert metrics.invalid_probability_mass == approx(0.7)
    assert metrics.invalid_target_rate == 0.375
    assert metrics.cnn_gradient_norm == 1.25
    assert metrics.gru_gradient_norm == 1.75
    assert metrics.transformer_gradient_norm == 2.25
