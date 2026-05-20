from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_model.paths import FINETUNING_CONFIG_PATH, PRETRAINING_CONFIG_PATH
from musak_model.tokens.schema import ScaleType
from musak_shared.files import load_yaml_config
from musak_shared.time_signature import validate_time_denominator


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


class GenerationEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    every_epochs: int = Field(default=5, ge=1)
    soft_sample_count: int = Field(default=4, ge=0)
    hard_sample_count: int = Field(default=4, ge=0)
    max_new_tokens: int = Field(default=256, ge=1)
    seed: int = 1729
    temperature: float = Field(default=1.0, gt=0)
    top_k: int | None = Field(default=32, ge=1)
    scale_root: int = Field(default=0, ge=0, lt=12)
    scale_type: ScaleType = ScaleType.MAJOR
    time_numerator: int = Field(default=4, ge=1)
    time_denominator: int = Field(default=4, ge=1)
    bar_count: int = Field(default=2, ge=1)
    minimum_duration_denominator: int | None = Field(default=16, ge=1)
    allow_dotted_durations: bool = True
    max_notes_per_hand: int | None = Field(default=5, ge=1)
    maximum_onset_span_semitones: int | None = Field(default=12, ge=0)
    maximum_pitch_gap_semitones: int | None = Field(default=12, ge=0)
    maximum_static_hand_span_degrees: int | None = Field(default=5, ge=1)

    @field_validator("minimum_duration_denominator")
    @classmethod
    def check_minimum_duration_denominator(cls, value: int | None) -> int | None:
        if value is not None:
            validate_time_denominator(value)

        return value

    @field_validator("time_denominator")
    @classmethod
    def check_time_denominator(cls, value: int) -> int:
        validate_time_denominator(value)
        return value


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimization: OptimizationConfig
    runtime: RuntimeConfig
    conditioning: TrainingConditioningConfig = TrainingConditioningConfig()
    checkpoints: CheckpointConfig
    mlflow: MlflowConfig = MlflowConfig()
    generation_evaluation: GenerationEvaluationConfig = GenerationEvaluationConfig()

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
