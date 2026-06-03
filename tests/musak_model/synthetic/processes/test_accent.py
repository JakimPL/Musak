import pytest
from numpy.random import Generator, default_rng
from pydantic import ValidationError

from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldSampler, draw_onset_mask
from musak_model.tokens.schema import Hand, ScaleType


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


def _sample_weights(
    sampler: AccentFieldSampler, *, bar_count: int, grid_count_per_bar: int, rng: Generator
) -> tuple[float, ...]:
    return sampler.sample_weights(
        bar_count=bar_count,
        grid_count_per_bar=grid_count_per_bar,
        scale_type=ScaleType.MAJOR,
        hand=Hand.RIGHT,
        rng=rng,
    )


def test_default_config_loads() -> None:
    config = AccentFieldConfig.load()

    assert config.envelope_basis_count > 0
    assert config.metric_gain >= 0


def test_sample_weights_is_deterministic_for_a_given_seed() -> None:
    sampler = AccentFieldSampler(config=AccentFieldConfig.load())

    first = _sample_weights(sampler, bar_count=4, grid_count_per_bar=16, rng=default_rng(0))
    second = _sample_weights(sampler, bar_count=4, grid_count_per_bar=16, rng=default_rng(0))

    assert first == second


def test_sample_weights_returns_only_weights_in_unit_interval() -> None:
    sampler = AccentFieldSampler(config=AccentFieldConfig.load())

    weights = _sample_weights(sampler, bar_count=4, grid_count_per_bar=16, rng=default_rng(0))

    assert len(weights) == 64
    assert all(isinstance(weight, float) for weight in weights)
    assert all(0.0 <= weight <= 1.0 for weight in weights)


def test_flat_config_produces_constant_weights() -> None:
    sampler = AccentFieldSampler(config=_config(baseline_logit=0.0, metric_gain=0.0, envelope_amplitude=0.0))

    weights = _sample_weights(sampler, bar_count=1, grid_count_per_bar=8, rng=default_rng(0))

    assert {round(weight, 12) for weight in weights} == {0.5}


def test_indispensability_orders_weights_by_metrical_strength() -> None:
    sampler = AccentFieldSampler(config=_config(metric_gain=2.0, envelope_amplitude=0.0))

    weights = _sample_weights(sampler, bar_count=1, grid_count_per_bar=16, rng=default_rng(0))

    assert weights[0] > weights[8] > weights[4]
    assert weights[4] > weights[2]
    assert weights[2] > weights[1]


def test_sample_weights_rejects_non_positive_dimensions() -> None:
    sampler = AccentFieldSampler(config=_config())

    with pytest.raises(ValueError, match="bar_count"):
        _sample_weights(sampler, bar_count=0, grid_count_per_bar=4, rng=default_rng(0))

    with pytest.raises(ValueError, match="grid_count_per_bar"):
        _sample_weights(sampler, bar_count=2, grid_count_per_bar=0, rng=default_rng(0))


def test_config_rejects_negative_metric_gain() -> None:
    with pytest.raises(ValidationError):
        _config(metric_gain=-0.1)


def _transition_count(mask: tuple[bool, ...]) -> int:
    return sum(1 for previous, current in zip(mask, mask[1:]) if previous != current)


def test_draw_onset_mask_is_deterministic_for_a_given_seed() -> None:
    weights = _sample_weights(
        AccentFieldSampler(config=AccentFieldConfig.load()), bar_count=4, grid_count_per_bar=16, rng=default_rng(1)
    )

    first = draw_onset_mask(weights, rng=default_rng(3))
    second = draw_onset_mask(weights, rng=default_rng(3))

    assert first == second


def test_draw_onset_mask_density_tracks_weights() -> None:
    dense = draw_onset_mask((0.9,) * 200, rng=default_rng(0))
    sparse = draw_onset_mask((0.1,) * 200, rng=default_rng(0))

    assert sum(dense) > 150
    assert sum(sparse) < 50


def test_clustered_weights_yield_clustered_onsets() -> None:
    half = 16
    clustered_weights = tuple(0.97 if index < half else 0.03 for index in range(2 * half))
    flat_weights = (0.5,) * (2 * half)

    clustered_mask = draw_onset_mask(clustered_weights, rng=default_rng(5))
    flat_mask = draw_onset_mask(flat_weights, rng=default_rng(5))

    assert _transition_count(clustered_mask) < _transition_count(flat_mask)
