from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from musak_model.paths import N_GRAM_ANALYSIS_CONFIG_PATH
from musak_shared.files import load_yaml_config

DEFAULT_FIGURE_COMMON_MASS_THRESHOLD: Final[float] = 0.80
DEFAULT_RHYTHM_MIN_N: Final[int] = 2
DEFAULT_RHYTHM_MAX_N: Final[int] = 4
DEFAULT_GRID_ALIGNMENT_DENOMINATORS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)
DEFAULT_STRONG_BEAT_OFFSETS: Final[tuple[Fraction, ...]] = (Fraction(0),)


class NGramAnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_n: int = Field(gt=0)
    max_n: int = Field(gt=0)
    limit_per_group: int | None = Field(default=None, gt=0)
    workers: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    figure_common_mass_threshold: float = Field(default=DEFAULT_FIGURE_COMMON_MASS_THRESHOLD, gt=0, le=1)
    rhythm_min_n: int = Field(default=DEFAULT_RHYTHM_MIN_N, gt=0)
    rhythm_max_n: int = Field(default=DEFAULT_RHYTHM_MAX_N, gt=0)
    grid_alignment_denominators: tuple[int, ...] = DEFAULT_GRID_ALIGNMENT_DENOMINATORS
    strong_beat_offsets: tuple[Fraction, ...] = DEFAULT_STRONG_BEAT_OFFSETS

    @model_validator(mode="after")
    def _validate_n_range(self) -> NGramAnalysisConfig:
        if self.max_n < self.min_n:
            raise ValueError("max_n must be greater than or equal to min_n")

        if self.rhythm_max_n < self.rhythm_min_n:
            raise ValueError("rhythm_max_n must be greater than or equal to rhythm_min_n")

        return self

    @field_validator("grid_alignment_denominators")
    @classmethod
    def _validate_grid_alignment_denominators(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("grid_alignment_denominators must not be empty")

        if any(denominator <= 0 for denominator in value):
            raise ValueError("grid_alignment_denominators must be positive")

        return value

    @field_validator("strong_beat_offsets")
    @classmethod
    def _validate_strong_beat_offsets(cls, value: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
        if not value:
            raise ValueError("strong_beat_offsets must not be empty")

        if any(offset < 0 for offset in value):
            raise ValueError("strong_beat_offsets must be non-negative")

        return value

    @classmethod
    def load(cls, path: Path = N_GRAM_ANALYSIS_CONFIG_PATH) -> NGramAnalysisConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
