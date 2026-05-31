from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from musak_model.paths import CALIBRATION_CONFIG_PATH
from musak_model.tokens.schema import ScaleType
from musak_shared.files import load_yaml_config
from musak_shared.misc import is_power_of_two


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    figure_root: Path
    output_path: Path
    scale_type: ScaleType
    scale_root: int = Field(ge=0)
    time_numerator: int = Field(gt=0)
    time_denominator: int = Field(gt=0)
    grid_count_per_bar: int | None = Field(default=None, gt=0)
    chord_resolution: int = Field(gt=0)
    bar_count: int = Field(gt=0)
    samples_per_config: int = Field(gt=0)
    min_n: int = Field(gt=0)
    max_n: int = Field(gt=0)
    self_transition_bias: float = Field(ge=0.0, le=1.0)
    commonness_bias: float = Field(ge=0.0)
    max_resample_retries: int = Field(gt=0)
    seed: int = Field(ge=0)
    lambda_curve: tuple[float, ...]
    lambda_harmonic: tuple[float, ...]
    lambda_accent: tuple[float, ...]
    target_total_variation_distance: float = Field(ge=0.0, le=1.0)

    @field_validator("chord_resolution")
    @classmethod
    def _validate_chord_resolution(cls, value: int) -> int:
        if not is_power_of_two(value):
            raise ValueError("chord_resolution must be a power of two note value (1 whole, 2 half, 4 quarter, ...)")

        return value

    @model_validator(mode="after")
    def _validate(self) -> CalibrationConfig:
        if self.max_n < self.min_n:
            raise ValueError("max_n must be greater than or equal to min_n")

        for name, values in (
            ("lambda_curve", self.lambda_curve),
            ("lambda_harmonic", self.lambda_harmonic),
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

    @property
    def grid_cells_per_bar(self) -> int:
        if self.grid_count_per_bar is not None:
            return self.grid_count_per_bar

        return self.time_numerator

    @classmethod
    def load(cls, path: Path = CALIBRATION_CONFIG_PATH) -> CalibrationConfig:
        return cls.model_validate(load_yaml_config(path))
