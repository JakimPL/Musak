from pathlib import Path

from musak_model.paths import (
    DEFAULT_FINETUNING_CHECKPOINT_DIRECTORY,
    DEFAULT_PRETRAINING_CHECKPOINT_DIRECTORY,
    FINETUNING_CONFIG_PATH,
    GENERATION_EVALUATION_CONFIG_PATH,
    PRETRAINING_CONFIG_PATH,
)
from musak_model.tokens.schema import ScaleType
from musak_model.training.config import GenerationEvaluationConfig
from scripts.evaluate_model import default_run_name, evaluation_stage_defaults, generation_config_with_overrides
from scripts.utils.train import TrainingStage


def _generation_config() -> GenerationEvaluationConfig:
    return GenerationEvaluationConfig(
        enabled=False,
        every_epochs=5,
        soft_sample_count=4,
        hard_sample_count=4,
        max_new_tokens=256,
        seed=1729,
        temperature=1.0,
        top_k=32,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        bar_count=2,
        minimum_duration_denominator=16,
        allow_dotted_durations=True,
        max_notes_per_hand=5,
        maximum_onset_span_semitones=12,
        maximum_pitch_gap_semitones=12,
        maximum_static_hand_span_degrees=5,
    )


def test_generation_config_loads_standalone_evaluation_yaml() -> None:
    config = GenerationEvaluationConfig.load(GENERATION_EVALUATION_CONFIG_PATH)

    assert config.enabled is True
    assert config.soft_sample_count == 16
    assert config.hard_sample_count == 64
    assert config.bar_count == 4
    assert config.max_new_tokens == 512
    assert config.top_k == 32


def test_generation_config_with_overrides_keeps_suite_shape_in_yaml() -> None:
    config = generation_config_with_overrides(
        _generation_config(),
        seed=123,
        temperature=0.8,
    )

    assert config.enabled is True
    assert config.soft_sample_count == 4
    assert config.hard_sample_count == 4
    assert config.bar_count == 2
    assert config.max_new_tokens == 256
    assert config.seed == 123
    assert config.temperature == 0.8
    assert config.top_k == 32


def test_pretrain_stage_defaults() -> None:
    defaults = evaluation_stage_defaults(TrainingStage.PRETRAINING)

    assert defaults.checkpoint_path == DEFAULT_PRETRAINING_CHECKPOINT_DIRECTORY / "best.pt"
    assert defaults.training_config_path == PRETRAINING_CONFIG_PATH
    assert defaults.mlflow_experiment_name == "musak-evaluate-pretrain"


def test_finetune_stage_defaults() -> None:
    defaults = evaluation_stage_defaults(TrainingStage.FINETUNING)

    assert defaults.checkpoint_path == DEFAULT_FINETUNING_CHECKPOINT_DIRECTORY / "best.pt"
    assert defaults.training_config_path == FINETUNING_CONFIG_PATH
    assert defaults.mlflow_experiment_name == "musak-evaluate-finetune"


def test_default_run_name_includes_stage_checkpoint_dataset_and_generation_shape() -> None:
    config = generation_config_with_overrides(
        _generation_config(),
        seed=None,
        temperature=None,
    )

    assert (
        default_run_name(
            stage=TrainingStage.PRETRAINING,
            data_dir=Path("PDMX"),
            checkpoint=Path("artifacts/checkpoints/pretraining/best.pt"),
            config=config,
        )
        == "eval-pretrain-best-PDMX-gen2b-4s4h-seed1729"
    )
    assert (
        default_run_name(
            stage=TrainingStage.FINETUNING,
            data_dir=Path("exercises"),
            checkpoint=Path("artifacts/checkpoints/finetuning/best.pt"),
            config=config,
        )
        == "eval-finetune-best-exercises-gen2b-4s4h-seed1729"
    )
