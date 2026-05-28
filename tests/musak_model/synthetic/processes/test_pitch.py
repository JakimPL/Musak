from statistics import fmean

import pytest
from numpy.random import default_rng
from pydantic import ValidationError

from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.tokens.schema import HAND_HOME_OCTAVES, Hand, ScaleType


def _config(
    *,
    arch_amplitude: float = 4.0,
    arch_basis_count: int = 3,
    arch_decay: float = 1.0,
    ou_theta: float = 0.2,
    ou_sigma: float = 1.0,
) -> RegisterCurveConfig:
    return RegisterCurveConfig(
        arch_basis_count=arch_basis_count,
        arch_amplitude=arch_amplitude,
        arch_decay=arch_decay,
        ou_theta=ou_theta,
        ou_sigma=ou_sigma,
    )


def test_default_config_loads() -> None:
    config = RegisterCurveConfig.load()

    assert config.arch_basis_count > 0
    assert 0 < config.ou_theta <= 1


def test_sample_is_deterministic_for_a_given_seed() -> None:
    sampler = RegisterCurveSampler(config=_config())

    first = sampler.sample(length=32, scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, rng=default_rng(42))
    second = sampler.sample(length=32, scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, rng=default_rng(42))

    assert first == second


def test_zero_amplitude_and_sigma_yield_constant_home_register() -> None:
    sampler = RegisterCurveSampler(
        config=_config(arch_amplitude=0.0, ou_sigma=0.0),
    )

    trajectory = sampler.sample(length=8, scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, rng=default_rng(0))

    expected = HAND_HOME_OCTAVES[Hand.RIGHT] * 7
    assert trajectory == (expected,) * 8


def test_hand_home_register_differs_between_hands() -> None:
    sampler = RegisterCurveSampler(config=_config(arch_amplitude=0.0, ou_sigma=0.0))

    right = sampler.sample(length=4, scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, rng=default_rng(1))
    left = sampler.sample(length=4, scale_type=ScaleType.MAJOR, hand=Hand.LEFT, rng=default_rng(1))

    assert right[0] == HAND_HOME_OCTAVES[Hand.RIGHT] * 7
    assert left[0] == HAND_HOME_OCTAVES[Hand.LEFT] * 7
    assert right != left


def test_ou_only_trajectory_is_anchored_near_home_register_on_average() -> None:
    sampler = RegisterCurveSampler(config=_config(arch_amplitude=0.0, ou_theta=0.5, ou_sigma=1.0))
    rng = default_rng(7)

    home = HAND_HOME_OCTAVES[Hand.RIGHT] * 7
    means = [fmean(sampler.sample(length=64, scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, rng=rng)) for _ in range(40)]

    assert abs(fmean(means) - home) < 1.0


def test_sample_rejects_non_positive_length() -> None:
    sampler = RegisterCurveSampler(config=_config())

    with pytest.raises(ValueError, match="length"):
        sampler.sample(length=0, scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, rng=default_rng(0))


def test_config_rejects_out_of_range_ou_theta() -> None:
    with pytest.raises(ValidationError):
        _config(ou_theta=0.0)

    with pytest.raises(ValidationError):
        _config(ou_theta=1.5)
