from math import sqrt

import numpy as np
import pytest
from numpy.random import default_rng

from musak_model.n_grams.profile.register.schema import RegisterStatisticsKey, RegisterSums
from musak_model.synthetic.fitting.register import (
    RegisterMoments,
    fit_register_config,
    fit_register_overrides,
    fit_register_overrides_from_statistics,
    register_moments_from_statistics,
)
from musak_model.synthetic.processes._basis import band_limited_random
from musak_model.synthetic.processes.pitch import RegisterCurveConfig
from musak_model.tokens.schema import Hand, ScaleType


def test_fit_register_config_maps_autocorrelation_and_residual_std_to_ou() -> None:
    config = fit_register_config(
        RegisterMoments(trend_std=0.0, residual_std=1.0, residual_lag1_autocorrelation=0.8),
        arch_basis_count=3,
        arch_decay=1.0,
    )

    assert config.ou_theta == pytest.approx(0.2)
    assert config.ou_sigma == pytest.approx(sqrt(2.0 * 0.2 - 0.2**2))
    assert config.arch_amplitude == 0.0


def test_fit_register_config_clamps_theta_into_the_valid_range() -> None:
    fully_correlated = fit_register_config(
        RegisterMoments(trend_std=0.0, residual_std=1.0, residual_lag1_autocorrelation=1.0),
        arch_basis_count=3,
        arch_decay=1.0,
    )
    anticorrelated = fit_register_config(
        RegisterMoments(trend_std=0.0, residual_std=1.0, residual_lag1_autocorrelation=-0.5),
        arch_basis_count=3,
        arch_decay=1.0,
    )

    assert 0.0 < fully_correlated.ou_theta <= 1.0
    assert anticorrelated.ou_theta == 1.0


def test_fitted_arch_amplitude_reproduces_the_target_trend_std() -> None:
    config = fit_register_config(
        RegisterMoments(trend_std=4.0, residual_std=0.0, residual_lag1_autocorrelation=0.0),
        arch_basis_count=3,
        arch_decay=1.0,
    )
    rng = default_rng(0)

    trajectory_stds = [
        float(
            np.std(band_limited_random(length=128, basis_count=3, amplitude=config.arch_amplitude, decay=1.0, rng=rng))
        )
        for _ in range(200)
    ]

    assert abs(float(np.mean(trajectory_stds)) - 4.0) < 0.5


def test_fit_register_overrides_keys_each_group() -> None:
    default = RegisterCurveConfig(arch_basis_count=3, arch_amplitude=4.0, arch_decay=1.0, ou_theta=0.2, ou_sigma=1.0)

    overrides = fit_register_overrides(
        {
            (ScaleType.MAJOR, Hand.RIGHT): RegisterMoments(
                trend_std=2.0, residual_std=1.0, residual_lag1_autocorrelation=0.5
            ),
            (ScaleType.MAJOR, Hand.LEFT): RegisterMoments(
                trend_std=1.0, residual_std=0.5, residual_lag1_autocorrelation=0.25
            ),
        },
        default=default,
    )

    by_group = {(override.scale_type, override.hand): override.config for override in overrides}
    assert by_group[(ScaleType.MAJOR, Hand.RIGHT)].ou_theta == pytest.approx(0.5)
    assert by_group[(ScaleType.MAJOR, Hand.LEFT)].ou_theta == pytest.approx(0.75)
    assert by_group[(ScaleType.MAJOR, Hand.RIGHT)].arch_basis_count == default.arch_basis_count


def test_register_moments_from_statistics_reduces_sums_to_moments() -> None:
    statistics = {
        RegisterStatisticsKey(scale_type=ScaleType.MAJOR.value, hand=Hand.RIGHT.value): RegisterSums(
            trend_square_sum=8.0, residual_square_sum=18.0, residual_lag_product_sum=9.0, element_count=2
        )
    }

    moments = register_moments_from_statistics(statistics)[(ScaleType.MAJOR, Hand.RIGHT)]

    assert moments.trend_std == pytest.approx(sqrt(8.0 / 2))
    assert moments.residual_std == pytest.approx(sqrt(18.0 / 2))
    assert moments.residual_lag1_autocorrelation == pytest.approx(9.0 / 18.0)


def test_register_moments_from_statistics_skips_degenerate_groups() -> None:
    statistics = {
        RegisterStatisticsKey(scale_type=ScaleType.MAJOR.value, hand=Hand.RIGHT.value): RegisterSums(
            trend_square_sum=0.0, residual_square_sum=0.0, residual_lag_product_sum=0.0, element_count=0
        ),
        RegisterStatisticsKey(scale_type=ScaleType.MAJOR.value, hand=Hand.LEFT.value): RegisterSums(
            trend_square_sum=1.0, residual_square_sum=0.0, residual_lag_product_sum=0.0, element_count=4
        ),
    }

    assert register_moments_from_statistics(statistics) == {}


def test_fit_register_overrides_from_statistics_keys_present_groups() -> None:
    default = RegisterCurveConfig(arch_basis_count=1, arch_amplitude=4.0, arch_decay=1.0, ou_theta=0.2, ou_sigma=1.0)
    statistics = {
        RegisterStatisticsKey(scale_type=ScaleType.MAJOR.value, hand=Hand.RIGHT.value): RegisterSums(
            trend_square_sum=2.0, residual_square_sum=4.0, residual_lag_product_sum=2.0, element_count=4
        )
    }

    overrides = fit_register_overrides_from_statistics(statistics, default=default)

    assert {(override.scale_type, override.hand) for override in overrides} == {(ScaleType.MAJOR, Hand.RIGHT)}
