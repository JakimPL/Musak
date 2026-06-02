from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, Field

from musak_model.paths import RHYTHMIC_DENSITY_CONFIG_PATH
from musak_model.synthetic.processes._basis import band_limited_random
from musak_shared.files import load_yaml_config


class RhythmicDensityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    amplitude: float = Field(ge=0.0)
    basis_count: int = Field(gt=0)
    decay: float = Field(ge=0.0)

    @classmethod
    def load(cls, path: Path = RHYTHMIC_DENSITY_CONFIG_PATH) -> RhythmicDensityConfig:
        return cls.model_validate(load_yaml_config(path))


@dataclass(frozen=True)
class RhythmicDensitySampler:
    config: RhythmicDensityConfig

    def sample(
        self,
        *,
        length: int,
        rng: Generator,
    ) -> tuple[float, ...]:
        if length <= 0:
            return ()

        offsets = band_limited_random(
            length=length,
            basis_count=self.config.basis_count,
            amplitude=self.config.amplitude,
            decay=self.config.decay,
            rng=rng,
        )
        return tuple(float(offset) for offset in offsets)
