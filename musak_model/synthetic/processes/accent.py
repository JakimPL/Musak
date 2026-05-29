from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import expit

from musak_model.paths import ACCENT_FIELD_CONFIG_PATH
from musak_model.synthetic.processes._basis import band_limited_random
from musak_shared.files import load_yaml_config


class AccentFieldConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_logit: float
    metric_gain: float = Field(ge=0)
    metric_exponent: float = Field(ge=0)
    envelope_basis_count: int = Field(gt=0)
    envelope_amplitude: float = Field(ge=0)
    envelope_decay: float = Field(ge=0)

    @classmethod
    def load(cls, path: Path = ACCENT_FIELD_CONFIG_PATH) -> AccentFieldConfig:
        return cls.model_validate(load_yaml_config(path))


@dataclass(frozen=True)
class AccentFieldSampler:
    config: AccentFieldConfig

    def sample_weights(
        self,
        *,
        bar_count: int,
        grid_count_per_bar: int,
        rng: Generator,
    ) -> tuple[float, ...]:
        if bar_count <= 0:
            raise ValueError("bar_count must be positive")

        if grid_count_per_bar <= 0:
            raise ValueError("grid_count_per_bar must be positive")

        total_cells = bar_count * grid_count_per_bar
        envelope = band_limited_random(
            length=total_cells,
            basis_count=self.config.envelope_basis_count,
            amplitude=self.config.envelope_amplitude,
            decay=self.config.envelope_decay,
            rng=rng,
        )
        indispensability = _indispensability_per_position(grid_count_per_bar)
        logits = _logits(
            indispensability_per_position=indispensability,
            envelope=envelope,
            bar_count=bar_count,
            baseline_logit=self.config.baseline_logit,
            metric_gain=self.config.metric_gain,
            metric_exponent=self.config.metric_exponent,
        )
        probabilities: NDArray[np.float64] = expit(logits)
        return tuple(float(weight) for weight in probabilities)


def _indispensability_per_position(grid_count_per_bar: int) -> NDArray[np.float64]:
    positions = np.arange(grid_count_per_bar)
    return np.gcd(positions, grid_count_per_bar).astype(np.float64) / grid_count_per_bar


def _logits(
    *,
    indispensability_per_position: NDArray[np.float64],
    envelope: NDArray[np.float64],
    bar_count: int,
    baseline_logit: float,
    metric_gain: float,
    metric_exponent: float,
) -> NDArray[np.float64]:
    tiled_indispensability = np.tile(indispensability_per_position, bar_count)
    return baseline_logit + metric_gain * tiled_indispensability**metric_exponent + envelope
