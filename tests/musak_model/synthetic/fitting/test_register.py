from fractions import Fraction
from math import sqrt
from pathlib import Path

import numpy as np
import pytest
from numpy.random import default_rng

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.synthetic.fitting.register import (
    RegisterMoments,
    fit_register_config,
    fit_register_overrides,
    register_moments,
    register_moments_from_sequences,
)
from musak_model.synthetic.processes._basis import band_limited_random
from musak_model.synthetic.processes.pitch import RegisterCurveConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import EndToken, Hand, HandToken, NoteToken, ScaleType, Token


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


def test_register_moments_separates_low_frequency_trend_from_residual() -> None:
    length = 64
    steps = np.arange(length)
    trend_signal = 4.0 * np.cos(np.pi * (steps + 0.5) / length)  # the lowest mid-cell DCT mode
    residual_signal = np.where(steps % 2 == 0, 1.0, -1.0)  # fastest alternation
    sequence = (trend_signal + residual_signal).tolist()

    moments = register_moments_from_sequences({(ScaleType.MAJOR, Hand.RIGHT): [sequence]}, arch_basis_count=1)[
        (ScaleType.MAJOR, Hand.RIGHT)
    ]

    assert moments.trend_std == pytest.approx(4.0 / sqrt(2.0), abs=0.05)
    assert moments.residual_std == pytest.approx(1.0, abs=0.05)
    assert moments.residual_lag1_autocorrelation == pytest.approx(-1.0, abs=0.05)


def test_register_moments_skips_groups_without_separable_residual() -> None:
    moments = register_moments_from_sequences({(ScaleType.MAJOR, Hand.RIGHT): [[5.0, 7.0]]}, arch_basis_count=3)

    assert moments == {}


def test_register_moments_reads_onset_register_per_hand(duration_vocabulary: DurationVocabulary) -> None:
    quarter = duration_vocabulary.require_duration_id(Fraction(1, 4))
    tokens: list[Token] = [HandToken(hand=Hand.RIGHT)]
    for octave_offset in (0, 1, 0, 1, 0, 1, 0, 1):
        tokens.append(NoteToken(degree=1, accidental=0, octave_offset=octave_offset, duration_id=quarter))
    tokens.append(EndToken())
    segment = Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            window_start_bar=0,
            source_file=Path("test"),
            difficulty_level=None,
        ),
    )

    moments = register_moments([segment], duration_vocabulary=duration_vocabulary, arch_basis_count=1)

    assert (ScaleType.MAJOR, Hand.RIGHT) in moments
    assert moments[(ScaleType.MAJOR, Hand.RIGHT)].residual_std > 0.0
