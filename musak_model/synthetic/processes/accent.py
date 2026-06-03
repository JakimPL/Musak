from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import expit

from musak_model.paths import ACCENT_FIELD_CONFIG_PATH
from musak_model.synthetic.processes._basis import band_limited_random
from musak_model.tokens.schema import Hand, ScaleType
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


class AccentFieldOverride(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scale_type: ScaleType
    hand: Hand
    config: AccentFieldConfig


@dataclass(frozen=True)
class AccentFieldSampler:
    config: AccentFieldConfig
    overrides: tuple[AccentFieldOverride, ...] = ()

    def sample_weights(
        self,
        *,
        bar_count: int,
        grid_count_per_bar: int,
        scale_type: ScaleType,
        hand: Hand,
        rng: Generator,
    ) -> tuple[float, ...]:
        if bar_count <= 0:
            raise ValueError("bar_count must be positive")

        if grid_count_per_bar <= 0:
            raise ValueError("grid_count_per_bar must be positive")

        config = self._config_for(scale_type, hand)
        envelope = band_limited_random(
            length=bar_count * grid_count_per_bar,
            basis_count=config.envelope_basis_count,
            amplitude=config.envelope_amplitude,
            decay=config.envelope_decay,
            rng=rng,
        )
        logits = _logits(
            indispensability=indispensability_per_position(grid_count_per_bar),
            envelope=envelope,
            bar_count=bar_count,
            baseline_logit=config.baseline_logit,
            metric_gain=config.metric_gain,
            metric_exponent=config.metric_exponent,
        )
        probabilities: NDArray[np.float64] = expit(logits)
        return tuple(float(weight) for weight in probabilities)

    def _config_for(self, scale_type: ScaleType, hand: Hand) -> AccentFieldConfig:
        for override in self.overrides:
            if override.scale_type == scale_type and override.hand == hand:
                return override.config

        return self.config


def draw_onset_mask(weights: Sequence[float], *, rng: Generator) -> tuple[bool, ...]:
    probabilities = np.asarray(weights, dtype=np.float64)
    draws = rng.random(probabilities.size)
    return tuple(bool(fired) for fired in draws < probabilities)


def indispensability_per_position(grid_count_per_bar: int) -> NDArray[np.float64]:
    positions = np.arange(grid_count_per_bar)
    return np.gcd(positions, grid_count_per_bar).astype(np.float64) / grid_count_per_bar


def metrical_weight_over_span(
    *,
    start_cell: Fraction,
    duration_cells: Fraction,
    metrical_position: int,
    indispensability: NDArray[np.float64],
) -> float:
    grid_count_per_bar = len(indispensability)
    span_end = start_cell + duration_cells
    position = start_cell
    weight = 0.0
    while position < span_end:
        cell = math.floor(position)
        segment_end = min(span_end, Fraction(cell + 1))
        bar_position = (metrical_position + cell) % grid_count_per_bar
        weight += float(indispensability[bar_position]) * float(segment_end - position)
        position = segment_end

    return weight


def _logits(
    *,
    indispensability: NDArray[np.float64],
    envelope: NDArray[np.float64],
    bar_count: int,
    baseline_logit: float,
    metric_gain: float,
    metric_exponent: float,
) -> NDArray[np.float64]:
    tiled_indispensability = np.tile(indispensability, bar_count)
    return baseline_logit + metric_gain * tiled_indispensability**metric_exponent + envelope
