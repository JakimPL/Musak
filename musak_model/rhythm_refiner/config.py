from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.paths import RHYTHM_REFINER_CONFIG_PATH
from musak_model.rhythm_refiner.schema import RhythmGridConfig
from musak_model.training.config import CheckpointConfig, MlflowConfig, OptimizationConfig, RuntimeConfig
from musak_shared.files import load_yaml_config


class RhythmRefinerDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_cells: int | None = Field(default=None, ge=1)


class RhythmRefinerMaskingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mask_probability: float = Field(gt=0.0, lt=1.0)
    seed: int = 1729


class RhythmRefinerModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_size: int = Field(ge=1)
    transformer_layers: int = Field(ge=1)
    attention_heads: int = Field(ge=1)
    feedforward_size: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    max_bar_count: int = Field(ge=1)
    max_cells_per_bar: int = Field(ge=1)
    max_distance_cells: int = Field(ge=1)
    max_time_numerator: int = Field(ge=1)
    max_time_denominator: int = Field(ge=1)


class RhythmRefinerLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    activity_weight: float = Field(ge=0.0)
    coactivity_weight: float = Field(ge=0.0)


class RhythmRefinerTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    grid: RhythmGridConfig
    data: RhythmRefinerDataConfig
    masking: RhythmRefinerMaskingConfig
    model: RhythmRefinerModelConfig
    loss: RhythmRefinerLossConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig
    checkpoints: CheckpointConfig
    mlflow: MlflowConfig = MlflowConfig()

    @classmethod
    def load(cls, path: Path = RHYTHM_REFINER_CONFIG_PATH) -> RhythmRefinerTrainingConfig:
        return cls.model_validate(load_yaml_config(path))
