import pytest
from numpy.random import default_rng
from pydantic import ValidationError

from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldSampler


def _config(
    *,
    baseline_logit: float = 0.0,
    metric_gain: float = 0.0,
    metric_exponent: float = 1.0,
    envelope_basis_count: int = 3,
    envelope_amplitude: float = 0.0,
    envelope_decay: float = 1.0,
) -> AccentFieldConfig:
    return AccentFieldConfig(
        baseline_logit=baseline_logit,
        metric_gain=metric_gain,
        metric_exponent=metric_exponent,
        envelope_basis_count=envelope_basis_count,
        envelope_amplitude=envelope_amplitude,
        envelope_decay=envelope_decay,
    )


def test_default_config_loads() -> None:
    config = AccentFieldConfig.load()

    assert config.envelope_basis_count > 0
    assert config.metric_gain >= 0


def test_sample_weights_is_deterministic_for_a_given_seed() -> None:
    sampler = AccentFieldSampler(config=AccentFieldConfig.load())

    first = sampler.sample_weights(bar_count=4, grid_count_per_bar=16, rng=default_rng(0))
    second = sampler.sample_weights(bar_count=4, grid_count_per_bar=16, rng=default_rng(0))

    assert first == second


def test_sample_weights_returns_only_weights_in_unit_interval() -> None:
    sampler = AccentFieldSampler(config=AccentFieldConfig.load())

    weights = sampler.sample_weights(bar_count=4, grid_count_per_bar=16, rng=default_rng(0))

    assert len(weights) == 64
    assert all(isinstance(weight, float) for weight in weights)
    assert all(0.0 <= weight <= 1.0 for weight in weights)


def test_flat_config_produces_constant_weights() -> None:
    sampler = AccentFieldSampler(config=_config(baseline_logit=0.0, metric_gain=0.0, envelope_amplitude=0.0))

    weights = sampler.sample_weights(bar_count=1, grid_count_per_bar=8, rng=default_rng(0))

    assert {round(weight, 12) for weight in weights} == {0.5}


def test_indispensability_orders_weights_by_metrical_strength() -> None:
    sampler = AccentFieldSampler(config=_config(metric_gain=2.0, envelope_amplitude=0.0))

    weights = sampler.sample_weights(bar_count=1, grid_count_per_bar=16, rng=default_rng(0))

    assert weights[0] > weights[8] > weights[4]
    assert weights[4] > weights[2]
    assert weights[2] > weights[1]


def test_sample_weights_rejects_non_positive_dimensions() -> None:
    sampler = AccentFieldSampler(config=_config())

    with pytest.raises(ValueError, match="bar_count"):
        sampler.sample_weights(bar_count=0, grid_count_per_bar=4, rng=default_rng(0))

    with pytest.raises(ValueError, match="grid_count_per_bar"):
        sampler.sample_weights(bar_count=2, grid_count_per_bar=0, rng=default_rng(0))


def test_config_rejects_negative_metric_gain() -> None:
    with pytest.raises(ValidationError):
        _config(metric_gain=-0.1)
