from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from musak_model.paths import CALIBRATION_CONFIG_PATH
from musak_model.tokens.schema import ScaleType
from musak_shared.files import load_yaml_config


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    figure_root: Path
    output_path: Path
    scale_type: ScaleType
    scale_root: int = Field(ge=0)
    time_numerator: int = Field(gt=0)
    time_denominator: int = Field(gt=0)
    bar_count: int = Field(gt=0)
    samples_per_config: int = Field(gt=0)
    min_n: int = Field(gt=0)
    max_n: int = Field(gt=0)
    self_transition_bias: float = Field(ge=0.0, le=1.0)
    commonness_bias: float = Field(ge=0.0)
    max_resample_retries: int = Field(gt=0)
    seed: int = Field(ge=0)
    lambda_curve: tuple[float, ...]
    lambda_harm: tuple[float, ...]
    lambda_accent: tuple[float, ...]

    @model_validator(mode="after")
    def _validate(self) -> CalibrationConfig:
        if self.max_n < self.min_n:
            raise ValueError("max_n must be greater than or equal to min_n")

        for name, values in (
            ("lambda_curve", self.lambda_curve),
            ("lambda_harm", self.lambda_harm),
            ("lambda_accent", self.lambda_accent),
        ):
            if not values:
                raise ValueError(f"{name} must be non-empty")

            if any(value < 0.0 for value in values):
                raise ValueError(f"{name} must be non-negative")

        return self

    @property
    def figure_lengths(self) -> tuple[int, ...]:
        return tuple(range(self.min_n, self.max_n + 1))

    @classmethod
    def load(cls, path: Path = CALIBRATION_CONFIG_PATH) -> CalibrationConfig:
        return cls.model_validate(load_yaml_config(path))
