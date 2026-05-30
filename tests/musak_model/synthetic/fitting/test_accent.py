from collections import Counter

import pytest
from scipy.special import expit

from musak_model.n_grams.profile.rhythm.schema import RhythmCountKey
from musak_model.synthetic.fitting.accent import (
    AccentMoments,
    accent_moments_from_rhythm_counts,
    fit_accent_config,
    fit_accent_overrides,
    fit_accent_overrides_from_rhythm_counts,
)
from musak_model.synthetic.processes.accent import AccentFieldConfig
from musak_model.tokens.schema import Hand, ScaleType

_LEVELS = (0.0625, 0.125, 0.25, 0.5, 1.0)


def _default() -> AccentFieldConfig:
    return AccentFieldConfig(
        baseline_logit=-0.5,
        metric_gain=2.0,
        metric_exponent=1.0,
        envelope_basis_count=3,
        envelope_amplitude=0.5,
        envelope_decay=1.0,
    )


def _moments_from_curve(*, baseline: float, gain: float, exponent: float) -> AccentMoments:
    rates = tuple(float(expit(baseline + gain * level**exponent)) for level in _LEVELS)
    return AccentMoments(
        indispensability=_LEVELS,
        onset_rate=rates,
        opportunity_count=tuple(1000.0 for _ in _LEVELS),
    )


def test_fit_accent_config_recovers_known_baseline_gain_and_exponent() -> None:
    config = fit_accent_config(_moments_from_curve(baseline=-2.0, gain=4.0, exponent=1.0), default=_default())

    assert config.baseline_logit == pytest.approx(-2.0, abs=1e-6)
    assert config.metric_gain == pytest.approx(4.0, abs=1e-6)
    assert config.metric_exponent == 1.0


def test_fit_accent_config_selects_a_nonlinear_exponent_when_it_fits_better() -> None:
    config = fit_accent_config(_moments_from_curve(baseline=-1.5, gain=3.0, exponent=2.0), default=_default())

    assert config.metric_exponent == 2.0


def test_fit_accent_config_keeps_envelope_from_default() -> None:
    default = _default()

    config = fit_accent_config(_moments_from_curve(baseline=-1.0, gain=2.0, exponent=1.0), default=default)

    assert config.envelope_basis_count == default.envelope_basis_count
    assert config.envelope_amplitude == default.envelope_amplitude
    assert config.envelope_decay == default.envelope_decay


def test_fit_accent_config_clamps_metric_gain_to_non_negative() -> None:
    config = fit_accent_config(_moments_from_curve(baseline=0.5, gain=-3.0, exponent=1.0), default=_default())

    assert config.metric_gain == 0.0


def test_fit_accent_overrides_keys_each_group() -> None:
    moments = _moments_from_curve(baseline=-2.0, gain=4.0, exponent=1.0)

    overrides = fit_accent_overrides(
        {(ScaleType.MAJOR, Hand.RIGHT): moments, (ScaleType.MAJOR, Hand.LEFT): moments},
        default=_default(),
    )

    assert {(override.scale_type, override.hand) for override in overrides} == {
        (ScaleType.MAJOR, Hand.RIGHT),
        (ScaleType.MAJOR, Hand.LEFT),
    }


def test_accent_moments_pool_occupancy_by_indispensability() -> None:
    counts = Counter(
        {
            _occupancy_key(cell=0): 8,
            _occupancy_key(cell=2): 5,
            _occupancy_key(cell=1): 2,
            _occupancy_key(cell=3): 1,
            _bar_total_key(): 10,
        }
    )

    moments = accent_moments_from_rhythm_counts(counts, grid_denominator=4)[(ScaleType.MAJOR, Hand.RIGHT)]

    by_level = dict(zip(moments.indispensability, moments.onset_rate, strict=True))
    opportunities = dict(zip(moments.indispensability, moments.opportunity_count, strict=True))
    assert by_level[1.0] == pytest.approx(0.8)  # cell 0 occupied in 8 of 10 bars
    assert by_level[0.5] == pytest.approx(0.5)  # cell 2 occupied in 5 of 10 bars
    assert by_level[0.25] == pytest.approx(3.0 / 20.0)  # cells 1 and 3 pooled over two bar-cell sets
    assert opportunities[0.25] == pytest.approx(20.0)


def test_fit_accent_overrides_from_rhythm_counts_keys_present_groups() -> None:
    counts = Counter(
        {
            _occupancy_key(cell=0): 9,
            _occupancy_key(cell=2): 4,
            _occupancy_key(cell=1): 1,
            _bar_total_key(): 10,
        }
    )

    overrides = fit_accent_overrides_from_rhythm_counts(counts, default=_default(), grid_denominator=4)

    assert {(override.scale_type, override.hand) for override in overrides} == {(ScaleType.MAJOR, Hand.RIGHT)}


def _occupancy_key(*, cell: int) -> RhythmCountKey:
    return RhythmCountKey(
        scale_type=ScaleType.MAJOR.value,
        time_signature="4/4",
        hand=Hand.RIGHT.value,
        kind="onset_position",
        parameter="4",
        value=str(cell),
    )


def _bar_total_key() -> RhythmCountKey:
    return RhythmCountKey(
        scale_type=ScaleType.MAJOR.value,
        time_signature="4/4",
        hand=Hand.RIGHT.value,
        kind="bar_total",
        parameter="",
        value="",
    )
