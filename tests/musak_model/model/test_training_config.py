from pathlib import Path

import pytest
from pydantic import ValidationError

from musak_model.training.config import StageTwoTrainingConfig, TrainingConfig


def test_training_config_loads_nested_stage_one_config() -> None:
    config = TrainingConfig.load()

    assert config.optimization.epochs == 25
    assert config.epochs == config.optimization.epochs
    assert config.runtime.device == "cuda"
    assert config.checkpoint_dir == Path("checkpoints/stage_one")
    assert not config.conditioning.use_conditioning
    assert config.mlflow.mlflow_experiment_name == "musak-stage-one"


def test_stage_two_config_loads_nested_checkpoint_config() -> None:
    config = StageTwoTrainingConfig.load()

    assert config.stage_one_checkpoint == Path("checkpoints/stage_one/best.pt")
    assert config.checkpoints.stage_one_checkpoint == config.stage_one_checkpoint
    assert config.conditioning.use_conditioning
    assert config.conditioning.use_structural_conditioning


def test_training_config_keeps_flat_constructor_compatibility() -> None:
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        weight_decay=0.0,
        num_workers=0,
        checkpoint_dir=Path("checkpoints"),
        device="cpu",
    )

    assert config.optimization.batch_size == 2
    assert config.runtime.device == "cpu"
    assert config.checkpoint_dir == Path("checkpoints")


def test_training_config_rejects_unknown_flat_field() -> None:
    with pytest.raises(ValidationError, match="unknown"):
        TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            weight_decay=0.0,
            num_workers=0,
            checkpoint_dir=Path("checkpoints"),
            unknown=True,
        )
