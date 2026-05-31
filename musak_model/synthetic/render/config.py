from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.paths import RENDER_CONFIG_PATH
from musak_shared.files import load_yaml_config


class RenderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    commonness_bias: float = Field(ge=0.0)
    lambda_curve: float = Field(ge=0.0)
    lambda_harmonic: float = Field(ge=0.0)
    lambda_accent: float = Field(ge=0.0)
    max_resample_retries: int = Field(gt=0)

    @classmethod
    def load(cls, path: Path = RENDER_CONFIG_PATH) -> RenderConfig:
        return cls.model_validate(load_yaml_config(path))
