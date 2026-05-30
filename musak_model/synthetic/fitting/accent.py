from collections.abc import Mapping
from dataclasses import dataclass
from math import log
from typing import Final

from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldOverride
from musak_model.tokens.schema import Hand, ScaleType

_RATE_EPSILON: Final[float] = 1e-6


@dataclass(frozen=True)
class AccentMoments:
    strong_onset_rate: float
    weak_onset_rate: float


def fit_accent_config(
    moments: AccentMoments,
    *,
    default: AccentFieldConfig,
) -> AccentFieldConfig:
    baseline_logit = _logit(moments.weak_onset_rate)
    metric_gain = max(0.0, _logit(moments.strong_onset_rate) - baseline_logit)
    return AccentFieldConfig(
        baseline_logit=baseline_logit,
        metric_gain=metric_gain,
        metric_exponent=default.metric_exponent,
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


def _logit(rate: float) -> float:
    clamped = min(1.0 - _RATE_EPSILON, max(_RATE_EPSILON, rate))
    return log(clamped / (1.0 - clamped))
