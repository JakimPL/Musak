from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.common.files import load_yaml_config
from musak_model.paths import STAGE_TWO_TRAINING_CONFIG_PATH, TRAINING_CONFIG_PATH


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_workers: int = Field(ge=0)
    device: str = "cpu"


class TrainingConditioningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    use_conditioning: bool = False
    use_structural_conditioning: bool = False


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_dir: Path
    resume_checkpoint: Path | None = None


class StageTwoCheckpointConfig(CheckpointConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_one_checkpoint: Path


class MlflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enable_mlflow: bool = True
    mlflow_experiment_name: str = "musak-stage-one"
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimization: OptimizationConfig
    runtime: RuntimeConfig
    conditioning: TrainingConditioningConfig = TrainingConditioningConfig()
    checkpoints: CheckpointConfig
    mlflow: MlflowConfig = MlflowConfig()

    @classmethod
    def load(cls, path: Path = TRAINING_CONFIG_PATH) -> TrainingConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)


class StageTwoTrainingConfig(TrainingConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoints: StageTwoCheckpointConfig

    @classmethod
    def load(cls, path: Path = STAGE_TWO_TRAINING_CONFIG_PATH) -> StageTwoTrainingConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
