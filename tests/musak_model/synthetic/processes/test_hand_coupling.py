from statistics import fmean

import pytest
from numpy.random import default_rng
from pydantic import ValidationError

from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig, HandCouplingSampler
from musak_model.tokens.schema import Hand


def _config(
    *,
    co_activity_strength: float = 0.5,
    activity_right: float = 0.5,
    activity_left: float = 0.5,
) -> HandCouplingConfig:
    return HandCouplingConfig(
        co_activity_strength=co_activity_strength,
        activity_right=activity_right,
        activity_left=activity_left,
    )


def test_default_config_loads() -> None:
    config = HandCouplingConfig.load()

    assert 0.0 <= config.co_activity_strength <= 1.0
    assert 0.0 <= config.activity_right <= 1.0
    assert 0.0 <= config.activity_left <= 1.0


def test_sample_gates_is_deterministic_for_a_given_seed() -> None:
    sampler = HandCouplingSampler(config=_config(co_activity_strength=0.7))

    first = sampler.sample_gates(cell_count=32, rng=default_rng(5))
    second = sampler.sample_gates(cell_count=32, rng=default_rng(5))

    assert first == second


def test_full_co_activity_makes_gates_coincide() -> None:
    sampler = HandCouplingSampler(config=_config(co_activity_strength=1.0, activity_right=0.6, activity_left=0.6))

    gates = sampler.sample_gates(cell_count=128, rng=default_rng(0))

    assert all(gate[Hand.RIGHT] == gate[Hand.LEFT] for gate in gates)


def test_zero_co_activity_makes_gates_disjoint_under_balanced_activity() -> None:
    sampler = HandCouplingSampler(config=_config(co_activity_strength=0.0, activity_right=0.5, activity_left=0.5))

    gates = sampler.sample_gates(cell_count=128, rng=default_rng(0))

    assert all(gate[Hand.RIGHT] != gate[Hand.LEFT] for gate in gates)


def test_marginal_activity_converges_to_configured_rate() -> None:
    sampler = HandCouplingSampler(config=_config(co_activity_strength=0.6, activity_right=0.7, activity_left=0.3))

    gates = sampler.sample_gates(cell_count=5000, rng=default_rng(7))

    right_rate = fmean(1.0 if gate[Hand.RIGHT] else 0.0 for gate in gates)
    left_rate = fmean(1.0 if gate[Hand.LEFT] else 0.0 for gate in gates)
    assert abs(right_rate - 0.7) < 0.03
    assert abs(left_rate - 0.3) < 0.03


def test_zero_activity_hand_never_plays() -> None:
    sampler = HandCouplingSampler(config=_config(activity_right=0.0))

    gates = sampler.sample_gates(cell_count=64, rng=default_rng(0))

    assert all(not gate[Hand.RIGHT] for gate in gates)


def test_sample_gates_rejects_non_positive_cell_count() -> None:
    sampler = HandCouplingSampler(config=_config())

    with pytest.raises(ValueError, match="cell_count"):
        sampler.sample_gates(cell_count=0, rng=default_rng(0))


def test_config_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        _config(co_activity_strength=1.5)

    with pytest.raises(ValidationError):
        _config(activity_right=-0.1)
