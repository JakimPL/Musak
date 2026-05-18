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
    assert "make train-pretrain" in output
    assert "make train-finetune" in output
    assert "make mlflow" in output
    assert "PRETRAIN_PROCESSED_DIR" in output
    assert "FINETUNE_PROCESSED_DIR" in output
    assert "MLFLOW_PORT" in output
    assert "OVERWRITE=1" in output


def test_make_process_uses_data_dir_and_processed_root() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/PDMX", "PROCESSED_ROOT=processed", "NUM_WORKERS=4")

    assert "scripts/process_dataset.py" in output
    assert '--data-dir "data/PDMX"' in output
    assert '--processed-dir "processed"' in output
    assert '--workers "4"' in output


def test_make_train_pretrain_uses_descriptive_variables() -> None:
    output = _make_dry_run(
        "train-pretrain",
        "PRETRAIN_DATA_DIR=data/PDMX",
        "PRETRAIN_PROCESSED_DIR=processed/PDMX",
        "PRETRAIN_EPOCHS=25",
        "PRETRAIN_DEVICE=cuda",
        "PRETRAIN_NUM_WORKERS=2",
        "OVERWRITE=1",
    )

    assert "scripts/train_stage_one.py" in output
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
        "PRETRAIN_CHECKPOINT=checkpoints/stage_one/best.pt",
    )

    assert output.index("scripts/train_stage_one.py") < output.index("scripts/train_stage_two.py")
    assert '--data-dir "data/PDMX"' in output
    assert '--processed-dir "processed/PDMX"' in output
    assert '--data-dir "data/Exercises"' in output
    assert '--processed-dir "processed/Exercises"' in output
    assert '--stage-one-checkpoint "checkpoints/stage_one/best.pt"' in output


def test_make_mlflow_starts_dashboard_with_configurable_address() -> None:
    output = _make_dry_run("mlflow", "MLFLOW_DIR=mlruns", "MLFLOW_HOST=0.0.0.0", "MLFLOW_PORT=5050")

    assert "uv run mlflow ui" in output
    assert '--backend-store-uri "file:mlruns"' in output
    assert '--host "0.0.0.0"' in output
    assert '--port "5050"' in output
