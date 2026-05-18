from pathlib import Path

import pytest
from pydantic import ValidationError

from musak_model.training.config import (
    CheckpointConfig,
    OptimizationConfig,
    RuntimeConfig,
    StageTwoTrainingConfig,
    TrainingConfig,
)


def test_training_config_loads_nested_stage_one_config() -> None:
    config = TrainingConfig.load()

    assert config.optimization.epochs == 25
    assert config.runtime.device == "cuda"
    assert config.checkpoints.checkpoint_dir == Path("checkpoints/stage_one")
    assert config.conditioning.use_time_signature
    assert config.conditioning.use_scale_type
    assert not config.conditioning.use_difficulty
    assert not config.conditioning.use_structural_conditioning
    assert config.mlflow.mlflow_experiment_name == "musak-stage-one"


def test_stage_two_config_loads_nested_checkpoint_config() -> None:
    config = StageTwoTrainingConfig.load()

    assert config.checkpoints.stage_one_checkpoint == Path("checkpoints/stage_one/best.pt")
    assert config.conditioning.use_time_signature
    assert config.conditioning.use_scale_type
    assert not config.conditioning.use_difficulty
    assert config.conditioning.use_structural_conditioning


def test_training_config_accepts_nested_constructor() -> None:
    config = TrainingConfig(
        optimization=OptimizationConfig(epochs=1, batch_size=2, learning_rate=0.001, weight_decay=0.0),
        runtime=RuntimeConfig(num_workers=0, device="cpu"),
        checkpoints=CheckpointConfig(checkpoint_dir=Path("checkpoints")),
    )

    assert config.optimization.batch_size == 2
    assert config.runtime.device == "cpu"
    assert config.checkpoints.checkpoint_dir == Path("checkpoints")


def test_training_config_rejects_flat_fields() -> None:
    with pytest.raises(ValidationError, match="optimization"):
        TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            weight_decay=0.0,
            num_workers=0,
            checkpoint_dir=Path("checkpoints"),
        )


def test_training_config_rejects_old_conditioning_field() -> None:
    with pytest.raises(ValidationError, match="use_conditioning"):
        TrainingConfig(
            optimization=OptimizationConfig(epochs=1, batch_size=2, learning_rate=0.001, weight_decay=0.0),
            runtime=RuntimeConfig(num_workers=0, device="cpu"),
            checkpoints=CheckpointConfig(checkpoint_dir=Path("checkpoints")),
            conditioning={"use_conditioning": True},
        )
