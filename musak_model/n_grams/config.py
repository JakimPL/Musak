from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from musak_model.paths import N_GRAM_ANALYSIS_CONFIG_PATH
from musak_shared.files import load_yaml_config


class FigureAnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_n: int = Field(gt=0)
    max_n: int = Field(gt=0)
    limit_per_group: int | None = Field(gt=0)
    common_mass_threshold: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _validate_range(self) -> FigureAnalysisConfig:
        if self.max_n < self.min_n:
            raise ValueError("max_n must be greater than or equal to min_n")

        return self


class RhythmAnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_n: int = Field(gt=0)
    max_n: int = Field(gt=0)
    grid_alignment_denominators: tuple[int, ...]
    strong_beat_offsets: tuple[Fraction, ...]

    @model_validator(mode="after")
    def _validate_range(self) -> RhythmAnalysisConfig:
        if self.max_n < self.min_n:
            raise ValueError("rhythm max_n must be greater than or equal to rhythm min_n")

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


class RegisterAnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    arch_basis_count: int = Field(gt=0)


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workers: int = Field(gt=0)
    batch_size: int = Field(gt=0)


class NGramAnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    figure: FigureAnalysisConfig
    rhythm: RhythmAnalysisConfig
    register: RegisterAnalysisConfig
    execution: ExecutionConfig

    @classmethod
    def load(cls, path: Path = N_GRAM_ANALYSIS_CONFIG_PATH) -> NGramAnalysisConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
