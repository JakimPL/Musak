from collections.abc import Mapping
from dataclasses import dataclass
from math import log
from typing import Final

import numpy as np
from numpy.typing import NDArray

from musak_model.n_grams.profile.rhythm.schema import RhythmCountCounter
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldOverride, indispensability_per_position
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.ratios import parse_ratio

_RATE_EPSILON: Final[float] = 1e-6
_EXPONENT_CANDIDATES: Final[tuple[float, ...]] = (0.5, 0.75, 1.0, 1.5, 2.0)

type _OnsetOccupancy = dict[tuple[ScaleType, Hand], dict[tuple[str, int], int]]
type _BarTotals = dict[tuple[ScaleType, Hand], dict[str, int]]


@dataclass(frozen=True)
class AccentMoments:
    indispensability: tuple[float, ...]
    onset_rate: tuple[float, ...]
    opportunity_count: tuple[float, ...]


@dataclass(frozen=True)
class _ExponentFit:
    baseline: float
    gain: float
    exponent: float
    weighted_sse: float


def accent_moments_from_rhythm_counts(
    counts: RhythmCountCounter,
    *,
    grid_denominator: int,
) -> dict[tuple[ScaleType, Hand], AccentMoments]:
    occupancy = _onset_occupancy(counts, grid_denominator=grid_denominator)
    bar_totals = _bar_totals(counts)
    moments: dict[tuple[ScaleType, Hand], AccentMoments] = {}
    for group, bars_by_time_signature in bar_totals.items():
        occupied_by_level, opportunities_by_level = _pool_by_indispensability(
            cell_occupancy=occupancy.get(group, {}),
            bars_by_time_signature=bars_by_time_signature,
            grid_denominator=grid_denominator,
        )
        levels = sorted(level for level, opportunities in opportunities_by_level.items() if opportunities > 0.0)
        if len(levels) < 2:
            continue

        moments[group] = AccentMoments(
            indispensability=tuple(levels),
            onset_rate=tuple(occupied_by_level[level] / opportunities_by_level[level] for level in levels),
            opportunity_count=tuple(opportunities_by_level[level] for level in levels),
        )

    return moments


def fit_accent_config(moments: AccentMoments, *, default: AccentFieldConfig) -> AccentFieldConfig:
    regressor_base = np.asarray(moments.indispensability, dtype=np.float64)
    targets = np.array([_logit(rate) for rate in moments.onset_rate], dtype=np.float64)
    opportunities = np.asarray(moments.opportunity_count, dtype=np.float64)
    fit = min(
        (
            _fit_for_exponent(exponent, regressor_base=regressor_base, targets=targets, opportunities=opportunities)
            for exponent in _EXPONENT_CANDIDATES
        ),
        key=lambda candidate: candidate.weighted_sse,
    )
    return AccentFieldConfig(
        baseline_logit=fit.baseline,
        metric_gain=max(0.0, fit.gain),
        metric_exponent=fit.exponent,
        envelope_basis_count=default.envelope_basis_count,
        envelope_amplitude=default.envelope_amplitude,
        envelope_decay=default.envelope_decay,
    )


def fit_accent_overrides(
    moments_by_group: Mapping[tuple[ScaleType, Hand], AccentMoments],
    *,
    default: AccentFieldConfig,
) -> tuple[AccentFieldOverride, ...]:
    return tuple(
        AccentFieldOverride(scale_type=scale_type, hand=hand, config=fit_accent_config(moments, default=default))
        for (scale_type, hand), moments in moments_by_group.items()
    )


def fit_accent_overrides_from_rhythm_counts(
    counts: RhythmCountCounter,
    *,
    default: AccentFieldConfig,
    grid_denominator: int,
) -> tuple[AccentFieldOverride, ...]:
    return fit_accent_overrides(
        accent_moments_from_rhythm_counts(counts, grid_denominator=grid_denominator), default=default
    )


def _onset_occupancy(counts: RhythmCountCounter, *, grid_denominator: int) -> _OnsetOccupancy:
    parameter = str(grid_denominator)
    occupancy: _OnsetOccupancy = {}
    for key, count in counts.items():
        if key.kind != "onset_position" or key.parameter != parameter:
            continue

        group = (ScaleType(key.scale_type), Hand(key.hand))
        occupancy.setdefault(group, {})[(key.time_signature, int(key.value))] = count

    return occupancy


def _bar_totals(counts: RhythmCountCounter) -> _BarTotals:
    bar_totals: _BarTotals = {}
    for key, count in counts.items():
        if key.kind != "bar_total":
            continue

        group = (ScaleType(key.scale_type), Hand(key.hand))
        bar_totals.setdefault(group, {})[key.time_signature] = count

    return bar_totals


def _pool_by_indispensability(
    *,
    cell_occupancy: dict[tuple[str, int], int],
    bars_by_time_signature: dict[str, int],
    grid_denominator: int,
) -> tuple[dict[float, float], dict[float, float]]:
    occupied_by_level: dict[float, float] = {}
    opportunities_by_level: dict[float, float] = {}
    for time_signature, bars in bars_by_time_signature.items():
        cells_per_bar = parse_ratio(time_signature) * grid_denominator
        if cells_per_bar.denominator != 1:
            continue

        grid_count = int(cells_per_bar)
        weights = indispensability_per_position(grid_count)
        for cell in range(grid_count):
            level = float(weights[cell])
            occupied_by_level[level] = occupied_by_level.get(level, 0.0) + cell_occupancy.get((time_signature, cell), 0)
            opportunities_by_level[level] = opportunities_by_level.get(level, 0.0) + bars

    return occupied_by_level, opportunities_by_level


def _fit_for_exponent(
    exponent: float,
    *,
    regressor_base: NDArray[np.float64],
    targets: NDArray[np.float64],
    opportunities: NDArray[np.float64],
) -> _ExponentFit:
    regressors = regressor_base**exponent
    baseline, gain = _weighted_linear_fit(regressors, targets, opportunities)
    weighted_sse = _weighted_sse(regressors, targets, opportunities, intercept=baseline, slope=gain)
    return _ExponentFit(baseline=baseline, gain=gain, exponent=exponent, weighted_sse=weighted_sse)


def _weighted_linear_fit(
    regressors: NDArray[np.float64],
    targets: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float]:
    total_weight = float(weights.sum())
    weighted_x = float((weights * regressors).sum())
    weighted_y = float((weights * targets).sum())
    weighted_xx = float((weights * regressors * regressors).sum())
    weighted_xy = float((weights * regressors * targets).sum())
    denominator = total_weight * weighted_xx - weighted_x**2
    if denominator == 0.0:
        return weighted_y / total_weight, 0.0

    slope = (total_weight * weighted_xy - weighted_x * weighted_y) / denominator
    intercept = (weighted_y - slope * weighted_x) / total_weight
    return intercept, slope


def _weighted_sse(
    regressors: NDArray[np.float64],
    targets: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    intercept: float,
    slope: float,
) -> float:
    residuals = targets - (intercept + slope * regressors)
    return float((weights * residuals**2).sum())


def _logit(rate: float) -> float:
    clamped = min(1.0 - _RATE_EPSILON, max(_RATE_EPSILON, rate))
    return log(clamped / (1.0 - clamped))
