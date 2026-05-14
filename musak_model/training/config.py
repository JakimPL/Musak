from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.common.files import load_yaml_config
from musak_model.paths import TRAINING_CONFIG_PATH


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)

    num_workers: int = Field(ge=0)
    device: str = "cpu"

    use_conditioning: bool = False

    checkpoint_dir: Path
    resume_checkpoint: Path | None = None

    enable_mlflow: bool = True
    mlflow_experiment_name: str = "musak-stage-one"
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None

    @classmethod
    def load(cls, path: Path = TRAINING_CONFIG_PATH) -> TrainingConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
