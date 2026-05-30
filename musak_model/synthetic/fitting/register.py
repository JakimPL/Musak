from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from typing import Final

from musak_model.n_grams.profile.register.schema import RegisterStatistics
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveOverride
from musak_model.tokens.schema import Hand, ScaleType

_MIN_OU_THETA: Final[float] = 1e-3


@dataclass(frozen=True)
class RegisterMoments:
    trend_std: float
    residual_std: float
    residual_lag1_autocorrelation: float


def register_moments_from_statistics(statistics: RegisterStatistics) -> dict[tuple[ScaleType, Hand], RegisterMoments]:
    moments: dict[tuple[ScaleType, Hand], RegisterMoments] = {}
    for key, sums in statistics.items():
        if sums.element_count == 0 or sums.residual_square_sum == 0.0:
            continue

        moments[(ScaleType(key.scale_type), Hand(key.hand))] = RegisterMoments(
            trend_std=sqrt(sums.trend_square_sum / sums.element_count),
            residual_std=sqrt(sums.residual_square_sum / sums.element_count),
            residual_lag1_autocorrelation=sums.residual_lag_product_sum / sums.residual_square_sum,
        )

    return moments


def fit_register_config(
    moments: RegisterMoments,
    *,
    arch_basis_count: int,
    arch_decay: float,
) -> RegisterCurveConfig:
    theta = _ou_theta_from_autocorrelation(moments.residual_lag1_autocorrelation)
    arch_amplitude = moments.trend_std / _arch_unit_std(arch_basis_count=arch_basis_count, arch_decay=arch_decay)
    return RegisterCurveConfig(
        arch_basis_count=arch_basis_count,
        arch_amplitude=arch_amplitude,
        arch_decay=arch_decay,
        ou_theta=theta,
        ou_sigma=_ou_sigma_from_stationary_std(moments.residual_std, theta),
    )


def fit_register_overrides(
    moments_by_group: Mapping[tuple[ScaleType, Hand], RegisterMoments],
    *,
    default: RegisterCurveConfig,
) -> tuple[RegisterCurveOverride, ...]:
    return tuple(
        RegisterCurveOverride(
            scale_type=scale_type,
            hand=hand,
            config=fit_register_config(
                moments, arch_basis_count=default.arch_basis_count, arch_decay=default.arch_decay
            ),
        )
        for (scale_type, hand), moments in moments_by_group.items()
    )


def fit_register_overrides_from_statistics(
    statistics: RegisterStatistics,
    *,
    default: RegisterCurveConfig,
) -> tuple[RegisterCurveOverride, ...]:
    return fit_register_overrides(register_moments_from_statistics(statistics), default=default)


def _ou_theta_from_autocorrelation(lag1_autocorrelation: float) -> float:
    return min(1.0, max(_MIN_OU_THETA, 1.0 - lag1_autocorrelation))


def _ou_sigma_from_stationary_std(stationary_std: float, theta: float) -> float:
    return stationary_std * sqrt(2.0 * theta - theta**2)


def _arch_unit_std(*, arch_basis_count: int, arch_decay: float) -> float:
    return sqrt(0.5 * sum(index ** (-2.0 * arch_decay) for index in range(1, arch_basis_count + 1)))
