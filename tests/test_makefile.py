import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _make_dry_run(*arguments: str) -> str:
    if shutil.which("make") is None:
        pytest.skip("make is not installed")

    result = subprocess.run(
        ["make", "-n", *arguments],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_make_install_uses_dev_extra_and_model_group() -> None:
    output = _make_dry_run("install")

    assert "uv sync --extra dev --group model" in output
    assert "uv run pre-commit install" in output


def test_make_help_documents_examples_and_variables() -> None:
    output = _make_dry_run("help")

    assert "Musak development commands" in output
    assert "make parse" in output
    assert "make tokenize" in output
    assert "make pretrain" in output
    assert "make finetune" in output
    assert "make mlflow" in output
    assert "PRETRAIN_PROCESSED_DIR" in output
    assert "FINETUNE_PROCESSED_DIR" in output
    assert "MLFLOW_PORT" in output
    assert "OVERWRITE=1" in output
    assert "PROCESS_OVERWRITE=1" in output
    assert "PROCESS_DISABLE_MLFLOW" in output
    assert "PROCESS_MLFLOW_EXPERIMENT" in output
    assert "PROCESS_DIFFICULTY_LABELS" in output
    assert "PROCESS_WHOLE_FILE_SEGMENTS" in output
    assert "PROFILE=1" in output
    assert "PROCESS_PROFILE=1" in output
    assert "FINETUNE_DIFFICULTY_LABELS" in output
    assert "FINETUNE_WHOLE_FILE_SEGMENTS" in output


def test_make_process_uses_data_dir_and_processed_root() -> None:
    output = _make_dry_run(
        "process",
        "DATA_DIR=data/PDMX",
        "PROCESSED_ROOT=processed",
        "NUM_WORKERS=4",
        "PROCESS_MLFLOW_RUN_NAME=process-test",
        "PROCESS_MLFLOW_TRACKING_URI=file:///tmp/mlruns",
        "PROCESS_DIFFICULTY_LABELS=data/PDMX/difficulty_labels.json",
        "PROCESS_WHOLE_FILE_SEGMENTS=1",
    )

    assert "scripts/process_dataset.py" in output
    assert '--stage "parse"' in output
    assert '--stage "tokenize"' in output
    assert '--data-dir "data/PDMX"' in output
    assert '--processed-dir "processed"' in output
    assert '--workers "4"' in output
    assert '--difficulty-labels "data/PDMX/difficulty_labels.json"' in output
    assert "--whole-file-segments" in output
    assert '--mlflow-experiment-name "musak-process"' in output
    assert '--mlflow-run-name "process-test"' in output
    assert '--mlflow-tracking-uri "file:///tmp/mlruns"' in output


def test_make_process_forwards_overwrite_flag() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/PDMX", "OVERWRITE=1")

    assert "--overwrite" in output


def test_make_process_supports_process_specific_overwrite_flag() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/PDMX", "PROCESS_OVERWRITE=1")

    assert "--overwrite" in output


def test_make_process_supports_overwite_typo_alias() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/PDMX", "OVERWITE=1")

    assert "--overwrite" in output


def test_make_process_can_disable_mlflow() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/PDMX", "PROCESS_DISABLE_MLFLOW=1")

    assert "--disable-mlflow" in output


def test_make_process_supports_profile_flag() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/PDMX", "PROFILE=1")

    assert "--profile" in output


def test_make_process_supports_process_specific_profile_flag() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/PDMX", "PROCESS_PROFILE=1")

    assert "--profile" in output


def test_make_parse_and_tokenize_expose_separate_stages() -> None:
    parse_output = _make_dry_run("parse", "DATA_DIR=data/PDMX")
    tokenize_output = _make_dry_run("tokenize", "DATA_DIR=data/PDMX")

    assert '--stage "parse"' in parse_output
    assert '--stage "tokenize"' in tokenize_output


def test_make_train_pretrain_uses_descriptive_variables() -> None:
    output = _make_dry_run(
        "pretrain",
        "PRETRAIN_DATA_DIR=data/PDMX",
        "PRETRAIN_PROCESSED_DIR=processed/PDMX",
        "PRETRAIN_EPOCHS=25",
        "PRETRAIN_DEVICE=cuda",
        "PRETRAIN_NUM_WORKERS=2",
        "OVERWRITE=1",
    )

    assert "scripts/pretrain.py" in output
    assert '--data-dir "data/PDMX"' in output
    assert '--processed-dir "processed/PDMX"' in output
    assert '--epochs "25"' in output
    assert '--device "cuda"' in output
    assert '--num-workers "2"' in output
    assert "--overwrite" in output


def test_make_train_runs_pretrain_then_finetune_with_distinct_datasets() -> None:
    output = _make_dry_run(
        "train",
        "PRETRAIN_DATA_DIR=data/PDMX",
        "PRETRAIN_PROCESSED_DIR=processed/PDMX",
        "FINETUNE_DATA_DIR=data/Exercises",
        "FINETUNE_PROCESSED_DIR=processed/Exercises",
        "FINETUNE_DIFFICULTY_LABELS=data/Exercises/difficulty_labels.json",
        "PRETRAIN_CHECKPOINT=checkpoints/pretraining/best.pt",
    )

    assert output.index("scripts/pretrain.py") < output.index("scripts/finetune.py")
    assert '--data-dir "data/PDMX"' in output
    assert '--processed-dir "processed/PDMX"' in output
    assert '--data-dir "data/Exercises"' in output
    assert '--processed-dir "processed/Exercises"' in output
    assert '--difficulty-labels "data/Exercises/difficulty_labels.json"' in output
    assert "--whole-file-segments" in output
    assert '--pretrain-checkpoint "checkpoints/pretraining/best.pt"' in output


def test_make_finetune_requires_difficulty_labels() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _make_dry_run(
            "finetune",
            "FINETUNE_DATA_DIR=data/Exercises",
            "FINETUNE_PROCESSED_DIR=processed/Exercises",
        )


def test_make_mlflow_starts_dashboard_with_configurable_address() -> None:
    output = _make_dry_run("mlflow", "MLFLOW_DIR=mlruns", "MLFLOW_HOST=0.0.0.0", "MLFLOW_PORT=5050")

    assert "uv run mlflow ui" in output
    assert '--backend-store-uri "file:mlruns"' in output
    assert '--host "0.0.0.0"' in output
    assert '--port "5050"' in output
