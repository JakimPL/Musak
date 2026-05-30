from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import lfilter

from musak_model.paths import REGISTER_CURVE_CONFIG_PATH
from musak_model.synthetic.processes._basis import band_limited_random
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.files import load_yaml_config


class RegisterCurveConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    arch_basis_count: int = Field(gt=0)
    arch_amplitude: float = Field(ge=0)
    arch_decay: float = Field(ge=0)
    ou_theta: float = Field(gt=0, le=1)
    ou_sigma: float = Field(ge=0)

    @classmethod
    def load(cls, path: Path = REGISTER_CURVE_CONFIG_PATH) -> RegisterCurveConfig:
        return cls.model_validate(load_yaml_config(path))


class RegisterCurveOverride(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scale_type: ScaleType
    hand: Hand
    config: RegisterCurveConfig


@dataclass(frozen=True)
class RegisterCurveSampler:
    config: RegisterCurveConfig
    overrides: tuple[RegisterCurveOverride, ...] = ()

    def sample(
        self,
        *,
        length: int,
        scale_type: ScaleType,
        hand: Hand,
        rng: Generator,
    ) -> tuple[int, ...]:
        if length <= 0:
            raise ValueError("length must be positive")

        config = self._config_for(scale_type, hand)
        arch = self._arch_trajectory(config, length=length, rng=rng)
        residual = self._residual_trajectory(config, length=length, rng=rng)
        return tuple(np.rint(arch + residual).astype(int).tolist())

    def _config_for(self, scale_type: ScaleType, hand: Hand) -> RegisterCurveConfig:
        for override in self.overrides:
            if override.scale_type == scale_type and override.hand == hand:
                return override.config

        return self.config

    @staticmethod
    def _arch_trajectory(config: RegisterCurveConfig, *, length: int, rng: Generator) -> NDArray[np.float64]:
        return band_limited_random(
            length=length,
            basis_count=config.arch_basis_count,
            amplitude=config.arch_amplitude,
            decay=config.arch_decay,
            rng=rng,
        )

    @staticmethod
    def _residual_trajectory(config: RegisterCurveConfig, *, length: int, rng: Generator) -> NDArray[np.float64]:
        residuals = np.zeros(length, dtype=np.float64)
        if length <= 1:
            return residuals

        innovations = rng.normal(loc=0.0, scale=config.ou_sigma, size=length - 1)
        pole = 1.0 - config.ou_theta
        residuals[1:] = lfilter([1.0], [1.0, -pole], innovations)
        return residuals
