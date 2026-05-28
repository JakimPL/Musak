from statistics import fmean

import pytest
from numpy.random import Generator, default_rng
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


def test_sample_is_deterministic_for_a_given_seed() -> None:
    sampler = AccentFieldSampler(config=AccentFieldConfig.load())

    first = sampler.sample(bar_count=4, grid_count_per_bar=16, rng=default_rng(0))
    second = sampler.sample(bar_count=4, grid_count_per_bar=16, rng=default_rng(0))

    assert first == second


def test_cell_count_equals_bar_count_times_grid_per_bar() -> None:
    sampler = AccentFieldSampler(config=_config())

    cells = sampler.sample(bar_count=3, grid_count_per_bar=8, rng=default_rng(0))

    assert len(cells) == 24
    assert {cell.bar_index for cell in cells} == {0, 1, 2}
    assert {cell.position for cell in cells} == set(range(8))


def test_flat_config_produces_constant_weights() -> None:
    sampler = AccentFieldSampler(config=_config(baseline_logit=0.0, metric_gain=0.0, envelope_amplitude=0.0))

    cells = sampler.sample(bar_count=1, grid_count_per_bar=8, rng=default_rng(0))

    weights = {round(cell.weight, 12) for cell in cells}
    assert weights == {0.5}


def test_indispensability_orders_weights_by_metrical_strength() -> None:
    sampler = AccentFieldSampler(config=_config(metric_gain=2.0, envelope_amplitude=0.0))

    cells = sampler.sample(bar_count=1, grid_count_per_bar=16, rng=default_rng(0))

    weights_by_position = {cell.position: cell.weight for cell in cells}
    assert weights_by_position[0] > weights_by_position[8] > weights_by_position[4]
    assert weights_by_position[4] > weights_by_position[2]
    assert weights_by_position[2] > weights_by_position[1]


def test_strong_metric_bias_concentrates_onsets_on_downbeats() -> None:
    sampler = AccentFieldSampler(config=_config(baseline_logit=-6.0, metric_gain=10.0, envelope_amplitude=0.0))
    rng = default_rng(0)

    downbeat_rate = fmean(
        1.0 if cell.onset else 0.0
        for cell in (cell for cell in _aggregate(sampler, rng, runs=40) if cell.position == 0)
    )
    offbeat_rate = fmean(
        1.0 if cell.onset else 0.0
        for cell in (cell for cell in _aggregate(sampler, rng, runs=40) if cell.position % 2 == 1)
    )

    assert downbeat_rate > offbeat_rate + 0.2


def test_sample_rejects_non_positive_dimensions() -> None:
    sampler = AccentFieldSampler(config=_config())

    with pytest.raises(ValueError, match="bar_count"):
        sampler.sample(bar_count=0, grid_count_per_bar=4, rng=default_rng(0))

    with pytest.raises(ValueError, match="grid_count_per_bar"):
        sampler.sample(bar_count=2, grid_count_per_bar=0, rng=default_rng(0))


def test_config_rejects_negative_metric_gain() -> None:
    with pytest.raises(ValidationError):
        _config(metric_gain=-0.1)


def _aggregate(sampler: AccentFieldSampler, rng: Generator, *, runs: int):
    for _ in range(runs):
        yield from sampler.sample(bar_count=1, grid_count_per_bar=16, rng=rng)
