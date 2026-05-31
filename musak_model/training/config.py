from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_model.model.config import ModelOutputMode
from musak_model.paths import DEFAULT_PRETRAINING_CHECKPOINT_DIRECTORY, FINETUNING_CONFIG_PATH, PRETRAINING_CONFIG_PATH
from musak_model.tokens.schema import ScaleType
from musak_shared.files import load_yaml_config
from musak_shared.time_signature import validate_time_denominator

DEFAULT_GENERATION_EVALUATION_SEED = 1729


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


class EventObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ModelOutputMode
    kind_weight: float = Field(ge=0.0)
    duration_weight: float = Field(ge=0.0)
    degree_weight: float = Field(ge=0.0)
    accidental_weight: float = Field(ge=0.0)
    octave_offset_weight: float = Field(ge=0.0)
    hand_weight: float = Field(ge=0.0)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_workers: int = Field(ge=0)
    device: str = "cpu"


class TrainingConditioningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    use_time_signature: bool
    use_scale_type: bool
    use_difficulty: bool
    use_structural_conditioning: bool
    use_validity_penalty: bool
    validity_penalty_weight: float = Field(ge=0.0)


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_directory: Path
    resume_checkpoint: Path | None = None
    save_all_epochs: bool = False


class FinetuningCheckpointConfig(CheckpointConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    save_all_epochs: bool = True
    pretraining_checkpoint: Path = DEFAULT_PRETRAINING_CHECKPOINT_DIRECTORY / "best.pt"


class MlflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enable_mlflow: bool = True
    mlflow_experiment_name: str = "musak-pretrain"
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None


class GenerationEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    every_epochs: int = Field(ge=1)
    soft_sample_count: int = Field(ge=0)
    hard_sample_count: int = Field(ge=0)
    max_new_tokens: int = Field(ge=1)
    seed: int = DEFAULT_GENERATION_EVALUATION_SEED
    temperature: float = Field(gt=0)
    top_k: int | None = Field(ge=1)
    scale_root: int = Field(ge=0, lt=12)
    scale_type: ScaleType
    time_numerator: int = Field(ge=1)
    time_denominator: int = Field(ge=1)
    bar_count: int = Field(ge=1)
    minimum_duration_denominator: int | None = Field(ge=1)
    allow_dotted_durations: bool
    max_notes_per_hand: int | None = Field(ge=1)
    maximum_onset_span_semitones: int | None = Field(ge=0)
    maximum_pitch_gap_semitones: int | None = Field(ge=0)
    maximum_static_hand_span_degrees: int | None = Field(ge=1)

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
    event_objective: EventObjectiveConfig
    runtime: RuntimeConfig
    conditioning: TrainingConditioningConfig
    checkpoints: CheckpointConfig
    mlflow: MlflowConfig = MlflowConfig()
    generation_evaluation: GenerationEvaluationConfig

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
