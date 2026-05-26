from pathlib import Path

import pytest
from pydantic import ValidationError

from musak_model.tokens.schema import ScaleType
from musak_model.training.config import (
    CheckpointConfig,
    GenerationEvaluationConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainingConditioningConfig,
    TrainingConfig,
)


def _conditioning_config() -> TrainingConditioningConfig:
    return TrainingConditioningConfig(
        use_time_signature=False,
        use_scale_type=False,
        use_difficulty=False,
        use_structural_conditioning=False,
        use_validity_penalty=False,
        validity_penalty_weight=0.05,
    )


def _generation_evaluation_config() -> GenerationEvaluationConfig:
    return GenerationEvaluationConfig(
        enabled=False,
        every_epochs=5,
        soft_sample_count=4,
        hard_sample_count=4,
        max_new_tokens=256,
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


def test_training_config_accepts_nested_constructor() -> None:
    config = TrainingConfig(
        optimization=OptimizationConfig(epochs=1, batch_size=2, learning_rate=0.001, weight_decay=0.0),
        runtime=RuntimeConfig(num_workers=0, device="cpu"),
        conditioning=_conditioning_config(),
        checkpoints=CheckpointConfig(checkpoint_directory=Path("checkpoints")),
        generation_evaluation=_generation_evaluation_config(),
    )

    assert config.optimization.batch_size == 2
    assert config.runtime.num_workers == 0
    assert config.runtime.device == "cpu"
    assert config.checkpoints.checkpoint_directory == Path("checkpoints")
    assert config.checkpoints.save_all_epochs is False


def test_training_config_rejects_flat_fields() -> None:
    with pytest.raises(ValidationError, match="optimization"):
        TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            weight_decay=0.0,
            num_workers=1,
            checkpoint_directory=Path("checkpoints"),
        )


def test_training_config_rejects_old_conditioning_field() -> None:
    with pytest.raises(ValidationError, match="use_conditioning"):
        TrainingConfig(
            optimization=OptimizationConfig(epochs=1, batch_size=2, learning_rate=0.001, weight_decay=0.0),
            runtime=RuntimeConfig(num_workers=1, device="cpu"),
            checkpoints=CheckpointConfig(checkpoint_directory=Path("checkpoints")),
            conditioning={"use_conditioning": True},
            generation_evaluation=_generation_evaluation_config(),
        )
