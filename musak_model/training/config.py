from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.paths import FINETUNING_CONFIG_PATH, PRETRAINING_CONFIG_PATH
from musak_shared.files import load_yaml_config


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

    use_time_signature: bool = False
    use_scale_type: bool = False
    use_difficulty: bool = False
    use_structural_conditioning: bool = False
    use_validity_penalty: bool = False
    validity_penalty_weight: float = Field(ge=0.0, default=0.05)


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_dir: Path
    resume_checkpoint: Path | None = None


class FinetuningCheckpointConfig(CheckpointConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pretraining_checkpoint: Path


class MlflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enable_mlflow: bool = True
    mlflow_experiment_name: str = "musak-pretrain"
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
    def load(cls, path: Path = PRETRAINING_CONFIG_PATH) -> TrainingConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)


class FinetuningTrainingConfig(TrainingConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoints: FinetuningCheckpointConfig

    @classmethod
    def load(cls, path: Path = FINETUNING_CONFIG_PATH) -> FinetuningTrainingConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
