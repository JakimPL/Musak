from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import NDArray

from musak_model.data.schema import Segment
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveOverride
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import note_diatonic_position
from musak_model.tokens.schema import Hand, ScaleType, scale_size_for_type

_MIN_OU_THETA: Final[float] = 1e-3

type RegisterSequences = Mapping[tuple[ScaleType, Hand], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class RegisterMoments:
    trend_std: float
    residual_std: float
    residual_lag1_autocorrelation: float


def register_moments(
    segments: Sequence[Segment],
    *,
    duration_vocabulary: DurationVocabulary,
    arch_basis_count: int,
) -> dict[tuple[ScaleType, Hand], RegisterMoments]:
    sequences_by_group: dict[tuple[ScaleType, Hand], list[Sequence[float]]] = {}
    for segment in segments:
        scale_size = scale_size_for_type(segment.scale_type)
        runs_by_hand = extract_hand_onset_runs(
            segment.tokens,
            duration_vocabulary=duration_vocabulary,
            time_numerator=segment.time_numerator,
            time_denominator=segment.time_denominator,
        )
        for hand, runs in runs_by_hand.items():
            sequence = [
                float(min(note_diatonic_position(note, scale_size=scale_size) for note in onset.notes))
                for run in runs
                for onset in run.onsets
            ]
            if sequence:
                sequences_by_group.setdefault((segment.scale_type, hand), []).append(sequence)

    return register_moments_from_sequences(sequences_by_group, arch_basis_count=arch_basis_count)


def register_moments_from_sequences(
    sequences_by_group: RegisterSequences,
    *,
    arch_basis_count: int,
) -> dict[tuple[ScaleType, Hand], RegisterMoments]:
    moments: dict[tuple[ScaleType, Hand], RegisterMoments] = {}
    for group, sequences in sequences_by_group.items():
        trend_square_sum = 0.0
        residual_square_sum = 0.0
        residual_lag_product_sum = 0.0
        element_count = 0
        minimum_length = arch_basis_count + 2
        for sequence in sequences:
            values = np.asarray(sequence, dtype=np.float64)
            if values.size < minimum_length:
                continue

            trend, residual = _trend_and_residual(values - values.mean(), arch_basis_count=arch_basis_count)
            trend_square_sum += float(trend @ trend)
            residual_square_sum += float(residual @ residual)
            residual_lag_product_sum += float(residual[:-1] @ residual[1:])
            element_count += residual.size

        if element_count == 0 or residual_square_sum == 0.0:
            continue

        moments[group] = RegisterMoments(
            trend_std=sqrt(trend_square_sum / element_count),
            residual_std=sqrt(residual_square_sum / element_count),
            residual_lag1_autocorrelation=residual_lag_product_sum / residual_square_sum,
        )

    return moments


def _trend_and_residual(
    centered: NDArray[np.float64], *, arch_basis_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    length = centered.size
    steps = np.arange(length)
    trend = np.zeros(length, dtype=np.float64)
    for index in range(1, min(arch_basis_count, length - 1) + 1):
        basis = np.cos(np.pi * index * (steps + 0.5) / length)
        coefficient = 2.0 / length * float(centered @ basis)
        trend += coefficient * basis

    return trend, centered - trend


def fit_register_config(
    moments: RegisterMoments,
    *,
    arch_basis_count: int,
    arch_decay: float,
) -> RegisterCurveConfig:
    theta = min(1.0, max(_MIN_OU_THETA, 1.0 - moments.residual_lag1_autocorrelation))
    sigma = moments.residual_std * sqrt(2.0 * theta - theta**2)
    arch_amplitude = moments.trend_std / _arch_unit_std(arch_basis_count=arch_basis_count, arch_decay=arch_decay)
    return RegisterCurveConfig(
        arch_basis_count=arch_basis_count,
        arch_amplitude=arch_amplitude,
        arch_decay=arch_decay,
        ou_theta=theta,
        ou_sigma=sigma,
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


def _arch_unit_std(*, arch_basis_count: int, arch_decay: float) -> float:
    return sqrt(0.5 * sum(index ** (-2.0 * arch_decay) for index in range(1, arch_basis_count + 1)))
