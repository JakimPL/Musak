from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.mlflow import MlflowRunConfig
from musak_model.paths import VALIDATION_CONFIG_PATH
from musak_model.tokens.schema import ScaleType
from musak_shared.files import load_yaml_config

_DEFAULT_MLFLOW = MlflowRunConfig(experiment_name="musak-validate-synthetic")


class SyntheticValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    figure_root: Path | None = None
    scales: tuple[ScaleType, ...] = Field(min_length=1)
    scale_root: int = Field(ge=0, lt=12)
    time_numerator: int = Field(gt=0)
    time_denominator: int = Field(gt=0)
    bar_count: int = Field(gt=0)
    harmonic_slot_denominator: int = Field(gt=0)
    prior_source: str
    samples_per_scale: int = Field(gt=0)
    base_seed: int = Field(ge=0)

    commonness_bias: float = Field(ge=0.0)
    lambda_curve: float = Field(ge=0.0)
    lambda_harmonic: float = Field(ge=0.0)
    lambda_accent: float = Field(ge=0.0)
    lambda_similarity: float = Field(ge=0.0)
    melodic_continuity: float = Field(ge=0.0, le=1.0)
    variation_budget: float = Field(ge=0.0, le=1.0)
    density_amplitude: float = Field(ge=0.0)
    density_basis_count: int = Field(gt=0)

    minimum_duration_denominator: int | None = None
    allow_dotted_durations: bool = True
    max_notes_per_hand: int | None = None
    maximum_onset_span_semitones: int | None = None
    maximum_pitch_gap_semitones: int | None = None
    maximum_static_hand_span_degrees: int | None = None

    sample_render_count: int = Field(ge=0)
    sweep: dict[str, tuple[float, ...]] | None = None
    mlflow: MlflowRunConfig = _DEFAULT_MLFLOW

    @classmethod
    def load(cls, path: Path = VALIDATION_CONFIG_PATH) -> SyntheticValidationConfig:
        return cls.model_validate(load_yaml_config(path))
