import argparse
from pathlib import Path

import pytest

from musak_model.model.config import ModelOutputMode
from musak_model.tokens.schema import ScaleType
from musak_model.training.config import (
    CheckpointConfig,
    EventObjectiveConfig,
    FinetuningCheckpointConfig,
    FinetuningTrainingConfig,
    GenerationEvaluationConfig,
    MlflowConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainingConditioningConfig,
    TrainingConfig,
)
from scripts.utils.train import (
    TrainingStage,
    resume_command,
    run_training_safely,
    should_skip_pretraining,
    validate_finetune_checkpoint,
)


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "data_dir": Path("data/PDMX"),
        "ingestion_config": Path("musak_model/configs/training/ingestion.yml"),
        "segmentation_config": Path("musak_model/configs/data/segmentation.yml"),
        "tokenization_config": Path("musak_model/configs/tokens/tokenization.yml"),
        "conditioning_config": Path("musak_model/configs/conditioning/conditioning.yml"),
        "training_config": Path("musak_model/configs/training/pretraining.yml"),
        "mlflow_db": Path("artifacts/mlflow/mlflow.db"),
        "log_level": "INFO",
        "no_progress": False,
        "learning_rate": None,
        "weight_decay": None,
        "validation_fraction": None,
        "split_seed": None,
        "window_bars": None,
        "stride_bars": None,
        "whole_file_segments": False,
        "save_all_epochs": False,
        "difficulty_labels": None,
        "mlflow_experiment_name": None,
        "mlflow_run_name": None,
        "overwrite": False,
        "resume_checkpoint": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _training_config(checkpoint_directory: Path, *, epochs: int = 25) -> TrainingConfig:
    return TrainingConfig(
        optimization=OptimizationConfig(epochs=epochs, batch_size=8, learning_rate=0.001, weight_decay=0.0),
        event_objective=_event_objective_config(),
        runtime=RuntimeConfig(num_workers=2, device="cuda"),
        checkpoints=CheckpointConfig(checkpoint_directory=checkpoint_directory),
        conditioning=_conditioning_config(use_time_signature=True, use_scale_type=True),
        mlflow=MlflowConfig(enable_mlflow=True),
        generation_evaluation=_generation_evaluation_config(),
    )


def _finetuning_config(
    checkpoint_directory: Path,
    *,
    pretraining_checkpoint: Path,
    save_all_epochs: bool = True,
) -> FinetuningTrainingConfig:
    return FinetuningTrainingConfig(
        optimization=OptimizationConfig(epochs=8, batch_size=8, learning_rate=0.001, weight_decay=0.0),
        event_objective=_event_objective_config(),
        runtime=RuntimeConfig(num_workers=4, device="cuda"),
        checkpoints=FinetuningCheckpointConfig(
            checkpoint_directory=checkpoint_directory,
            pretraining_checkpoint=pretraining_checkpoint,
            save_all_epochs=save_all_epochs,
        ),
        conditioning=_conditioning_config(),
        generation_evaluation=_generation_evaluation_config(),
    )


def _conditioning_config(
    *,
    use_time_signature: bool = False,
    use_scale_type: bool = False,
) -> TrainingConditioningConfig:
    return TrainingConditioningConfig(
        use_time_signature=use_time_signature,
        use_scale_type=use_scale_type,
        use_difficulty=False,
        use_structural_conditioning=False,
        use_validity_penalty=False,
        validity_penalty_weight=0.05,
    )


def _event_objective_config() -> EventObjectiveConfig:
    return EventObjectiveConfig(
        mode=ModelOutputMode.FLAT,
        kind_weight=1.0,
        duration_weight=1.0,
        degree_weight=1.0,
        accidental_weight=1.0,
        octave_offset_weight=1.0,
        hand_weight=1.0,
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


def test_resume_command_for_pretraining_copies_resolved_arguments(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretraining" / "latest.pt"
    config = _training_config(checkpoint.parent, epochs=25)

    command = resume_command(
        stage=TrainingStage.PRETRAINING,
        args=_args(no_progress=True),
        training_config=config,
        checkpoint_path=checkpoint,
    )

    assert "uv run python scripts/pretrain.py" in command
    assert "--data-dir data/PDMX" in command
    assert f"--checkpoint-dir {checkpoint.parent}" in command
    assert f"--resume-checkpoint {checkpoint}" in command
    assert "--epochs 25" in command
    assert "--device cuda" in command
    assert "--num-workers 2" in command
    assert "--use-conditioning" not in command
    assert "--no-progress" in command


def test_resume_command_for_finetuning_includes_pretrain_checkpoint(tmp_path: Path) -> None:
    latest_checkpoint = tmp_path / "finetuning" / "latest.pt"
    pretrain_checkpoint = tmp_path / "pretraining" / "best.pt"
    config = _finetuning_config(latest_checkpoint.parent, pretraining_checkpoint=pretrain_checkpoint)

    command = resume_command(
        stage=TrainingStage.FINETUNING,
        args=_args(training_config=Path("musak_model/configs/training/finetuning.yml")),
        training_config=config,
        checkpoint_path=latest_checkpoint,
    )

    assert "uv run python scripts/finetune.py" in command
    assert f"--resume-checkpoint {latest_checkpoint}" in command
    assert "--epochs 8" in command
    assert f"--pretrain-checkpoint {pretrain_checkpoint}" in command


def test_resume_command_preserves_save_all_epochs(tmp_path: Path) -> None:
    latest_checkpoint = tmp_path / "finetuning" / "latest.pt"
    pretrain_checkpoint = tmp_path / "pretraining" / "best.pt"
    config = _finetuning_config(latest_checkpoint.parent, pretraining_checkpoint=pretrain_checkpoint)

    command = resume_command(
        stage=TrainingStage.FINETUNING,
        args=_args(training_config=Path("musak_model/configs/training/finetuning.yml")),
        training_config=config,
        checkpoint_path=latest_checkpoint,
    )

    assert "--save-all-epochs" in command


def test_keyboard_interrupt_prints_resume_command_when_latest_checkpoint_exists(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "pretraining" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    config = _training_config(checkpoint.parent)

    with pytest.raises(SystemExit) as exit_info:
        run_training_safely(
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            stage=TrainingStage.PRETRAINING,
            args=_args(),
            training_config=config,
        )

    assert exit_info.value.code == 130
    output = capsys.readouterr().out
    assert "Training interrupted." in output
    assert f"--resume-checkpoint {checkpoint}" in output


def test_keyboard_interrupt_reports_missing_latest_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _training_config(tmp_path / "pretraining")

    with pytest.raises(SystemExit) as exit_info:
        run_training_safely(
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            stage=TrainingStage.PRETRAINING,
            args=_args(),
            training_config=config,
        )

    assert exit_info.value.code == 130
    output = capsys.readouterr().out
    assert "No latest checkpoint exists yet" in output


def test_pretraining_checkpoint_prompt_skips_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest_checkpoint = tmp_path / "pretraining" / "latest.pt"
    latest_checkpoint.parent.mkdir(parents=True)
    latest_checkpoint.write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _: "")

    assert should_skip_pretraining(args=_args(), training_config=_training_config(latest_checkpoint.parent))


def test_pretraining_checkpoint_prompt_runs_when_user_confirms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best_checkpoint = tmp_path / "pretraining" / "best.pt"
    best_checkpoint.parent.mkdir(parents=True)
    best_checkpoint.write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _: "yes")

    assert not should_skip_pretraining(args=_args(), training_config=_training_config(best_checkpoint.parent))


def test_pretraining_overwrite_bypasses_checkpoint_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest_checkpoint = tmp_path / "pretraining" / "latest.pt"
    latest_checkpoint.parent.mkdir(parents=True)
    latest_checkpoint.write_text("checkpoint", encoding="utf-8")

    def fail_input(_: str) -> str:
        raise AssertionError("input should not be called")

    monkeypatch.setattr("builtins.input", fail_input)

    assert not should_skip_pretraining(
        args=_args(overwrite=True),
        training_config=_training_config(latest_checkpoint.parent),
    )


def test_finetune_requires_pretrain_checkpoint(tmp_path: Path) -> None:
    config = _finetuning_config(
        tmp_path / "finetuning",
        pretraining_checkpoint=tmp_path / "pretraining" / "best.pt",
    )

    with pytest.raises(FileNotFoundError, match="Stage-one pretrain checkpoint"):
        validate_finetune_checkpoint(config)


def test_finetune_accepts_existing_pretrain_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretraining" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    config = _finetuning_config(tmp_path / "finetuning", pretraining_checkpoint=checkpoint)

    validate_finetune_checkpoint(config)
