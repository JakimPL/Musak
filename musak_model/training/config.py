from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_config(cls, value: Any) -> Any:
        section_keys = {"optimization", "runtime", "conditioning", "checkpoints", "mlflow"}
        if not isinstance(value, dict) or section_keys.intersection(value):
            return value

        flat_keys = {
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "num_workers",
            "device",
            "use_conditioning",
            "use_structural_conditioning",
            "checkpoint_dir",
            "resume_checkpoint",
            "stage_one_checkpoint",
            "enable_mlflow",
            "mlflow_experiment_name",
            "mlflow_run_name",
            "mlflow_tracking_uri",
        }
        extra_values = {key: nested_value for key, nested_value in value.items() if key not in flat_keys}
        return {
            "optimization": _take_existing(value, "epochs", "batch_size", "learning_rate", "weight_decay"),
            "runtime": _take_existing(value, "num_workers", "device"),
            "conditioning": _take_existing(value, "use_conditioning", "use_structural_conditioning"),
            "checkpoints": _take_existing(value, "checkpoint_dir", "resume_checkpoint", "stage_one_checkpoint"),
            "mlflow": _take_existing(
                value,
                "enable_mlflow",
                "mlflow_experiment_name",
                "mlflow_run_name",
                "mlflow_tracking_uri",
            ),
            **extra_values,
        }

    @classmethod
    def load(cls, path: Path = TRAINING_CONFIG_PATH) -> TrainingConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)

    @property
    def epochs(self) -> int:
        return self.optimization.epochs

    @property
    def batch_size(self) -> int:
        return self.optimization.batch_size

    @property
    def learning_rate(self) -> float:
        return self.optimization.learning_rate

    @property
    def weight_decay(self) -> float:
        return self.optimization.weight_decay

    @property
    def num_workers(self) -> int:
        return self.runtime.num_workers

    @property
    def device(self) -> str:
        return self.runtime.device

    @property
    def use_conditioning(self) -> bool:
        return self.conditioning.use_conditioning

    @property
    def use_structural_conditioning(self) -> bool:
        return self.conditioning.use_structural_conditioning

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoints.checkpoint_dir

    @property
    def resume_checkpoint(self) -> Path | None:
        return self.checkpoints.resume_checkpoint

    @property
    def enable_mlflow(self) -> bool:
        return self.mlflow.enable_mlflow

    @property
    def mlflow_experiment_name(self) -> str:
        return self.mlflow.mlflow_experiment_name

    @property
    def mlflow_run_name(self) -> str | None:
        return self.mlflow.mlflow_run_name

    @property
    def mlflow_tracking_uri(self) -> str | None:
        return self.mlflow.mlflow_tracking_uri


class StageTwoTrainingConfig(TrainingConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoints: StageTwoCheckpointConfig

    @classmethod
    def load(cls, path: Path = STAGE_TWO_TRAINING_CONFIG_PATH) -> StageTwoTrainingConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)

    @property
    def stage_one_checkpoint(self) -> Path:
        return self.checkpoints.stage_one_checkpoint


def _take_existing(values: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: values[key] for key in keys if key in values}
