from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm

from musak_model.paths import HAND_COUPLING_CONFIG_PATH
from musak_model.tokens.schema import Hand
from musak_shared.files import load_yaml_config


class HandCouplingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    co_activity_strength: float = Field(ge=0.0, le=1.0)
    activity_right: float = Field(ge=0.0, le=1.0)
    activity_left: float = Field(ge=0.0, le=1.0)

    @classmethod
    def load(cls, path: Path = HAND_COUPLING_CONFIG_PATH) -> HandCouplingConfig:
        return cls.model_validate(load_yaml_config(path))


@dataclass(frozen=True)
class HandCouplingSampler:
    config: HandCouplingConfig

    def sample_gates(
        self,
        *,
        cell_count: int,
        rng: Generator,
    ) -> tuple[dict[Hand, bool], ...]:
        if cell_count <= 0:
            raise ValueError("cell_count must be positive")

        correlation = 2.0 * self.config.co_activity_strength - 1.0
        covariance = [[1.0, correlation], [correlation, 1.0]]
        draws = rng.multivariate_normal(mean=[0.0, 0.0], cov=covariance, size=cell_count)

        right_threshold = float(norm.ppf(1.0 - self.config.activity_right))
        left_threshold = float(norm.ppf(1.0 - self.config.activity_left))
        right_active = draws[:, 0] > right_threshold
        left_active = draws[:, 1] > left_threshold
        return tuple(
            {Hand.RIGHT: bool(is_right_active), Hand.LEFT: bool(is_left_active)}
            for is_right_active, is_left_active in zip(right_active, left_active, strict=True)
        )
