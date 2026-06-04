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


def test_make_help_lists_main_targets() -> None:
    output = _make_dry_run("help")

    assert "Musak development commands" in output
    for target in ("app", "process", "pretrain", "finetune", "train", "mlflow", "notebook-<name>"):
        assert f"make {target}" in output


def test_make_process_uses_data_dir_and_processing_options() -> None:
    output = _make_dry_run(
        "process",
        "DATA_DIR=data/sample-dataset",
        "NUM_WORKERS=4",
        "PROCESS_MLFLOW_RUN_NAME=process-test",
        "PROCESS_MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow.db",
        "PROCESSING_CONFIG=musak_model/configs/data/processing.yml",
        "PROCESS_DIFFICULTY_LABELS=data/sample-difficulty.json",
        "PROCESS_WHOLE_FILE_SEGMENTS=1",
        "ANALYSIS_CONFIG=musak_model/configs/analysis/n_grams.yml",
        "ANALYSIS_OUTPUT=artifacts/analysis/sample-figures.csv",
        "ANALYSIS_NO_PROGRESS=1",
    )

    assert "scripts/process_dataset.py" in output
    assert '--stage "process"' in output
    assert '--stage "parse"' not in output
    assert '--stage "tokenize"' not in output
    assert '--data-dir "data/sample-dataset"' in output
    assert '--processing-config "musak_model/configs/data/processing.yml"' in output
    assert '--workers "4"' in output
    assert '--difficulty-labels "data/sample-difficulty.json"' in output
    assert "--whole-file-segments" in output
    assert '--mlflow-run-name "process-test"' in output
    assert '--mlflow-tracking-uri "sqlite:////tmp/mlflow.db"' in output
    assert '--analysis-config "musak_model/configs/analysis/n_grams.yml"' in output
    assert '--analysis-output "artifacts/analysis/sample-figures.csv"' in output
    assert "--no-progress" in output


def test_make_process_forwards_overwrite_flag() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/sample-dataset", "OVERWRITE=1")

    assert "--overwrite" in output


def test_make_process_supports_process_specific_overwrite_flag() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/sample-dataset", "PROCESS_OVERWRITE=1")

    assert "--overwrite" in output


def test_make_process_supports_overwite_typo_alias() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/sample-dataset", "OVERWITE=1")

    assert "--overwrite" in output


def test_make_process_can_disable_mlflow() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/sample-dataset", "PROCESS_DISABLE_MLFLOW=1")

    assert "--disable-mlflow" in output


def test_make_process_can_skip_figure_analysis() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/sample-dataset", "PROCESS_SKIP_FIGURE_ANALYSIS=1")

    assert "--skip-figure-analysis" in output


def test_make_process_supports_profile_flag() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/sample-dataset", "PROFILE=1")

    assert "--profile" in output


def test_make_process_supports_process_specific_profile_flag() -> None:
    output = _make_dry_run("process", "DATA_DIR=data/sample-dataset", "PROCESS_PROFILE=1")

    assert "--profile" in output


def test_make_parse_and_tokenize_expose_separate_stages() -> None:
    parse_output = _make_dry_run("parse", "DATA_DIR=data/sample-dataset")
    tokenize_output = _make_dry_run("tokenize", "DATA_DIR=data/sample-dataset")

    assert '--stage "parse"' in parse_output
    assert '--stage "tokenize"' in tokenize_output


def test_make_analyze_n_grams_uses_dataset_and_analysis_variables() -> None:
    output = _make_dry_run(
        "analyze-n-grams",
        "DATA_DIR=data/sample-dataset",
        "ANALYSIS_CONFIG=musak_model/configs/analysis/n_grams.yml",
        "ANALYSIS_OUTPUT=artifacts/analysis/sample-figures.csv",
        "ANALYSIS_ENCODED_DIRECTORY=artifacts/processed/sample-dataset/encoded/abc",
        "ANALYSIS_NO_PROGRESS=1",
    )

    assert "scripts/extract_figures.py" in output
    assert '--data-dir "data/sample-dataset"' in output
    assert '--analysis-config "musak_model/configs/analysis/n_grams.yml"' in output
    assert '--output "artifacts/analysis/sample-figures.csv"' in output
    assert '--encoded-directory "artifacts/processed/sample-dataset/encoded/abc"' in output
    assert "--no-progress" in output


def test_make_train_pretrain_uses_descriptive_variables() -> None:
    output = _make_dry_run(
        "pretrain",
        "PRETRAIN_DATA_DIR=data/sample-dataset",
        "PRETRAIN_EPOCHS=25",
        "PRETRAIN_DEVICE=cuda",
        "PRETRAIN_NUM_WORKERS=2",
        "OVERWRITE=1",
    )

    assert "scripts/pretrain.py" in output
    assert '--data-dir "data/sample-dataset"' in output
    assert '--epochs "25"' in output
    assert '--device "cuda"' in output
    assert '--num-workers "2"' in output
    assert "--overwrite" in output


def test_make_train_runs_pretrain_then_finetune_with_distinct_datasets() -> None:
    output = _make_dry_run(
        "train",
        "PRETRAIN_DATA_DIR=data/sample-dataset",
        "FINETUNE_DATA_DIR=data/finetuning-dataset",
        "FINETUNE_DIFFICULTY_LABELS=data/finetuning-difficulty.json",
        "PRETRAIN_CHECKPOINT=custom/pretraining.pt",
    )

    assert output.index("scripts/pretrain.py") < output.index("scripts/finetune.py")
    assert '--data-dir "data/sample-dataset"' in output
    assert '--data-dir "data/finetuning-dataset"' in output
    assert '--difficulty-labels "data/finetuning-difficulty.json"' in output
    assert "--whole-file-segments" in output
    assert '--pretrain-checkpoint "custom/pretraining.pt"' in output


def test_make_finetune_requires_difficulty_labels() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _make_dry_run(
            "finetune",
            "FINETUNE_DATA_DIR=data/finetuning-dataset",
        )


def test_make_evaluate_pretrain_dispatches_shared_evaluator() -> None:
    output = _make_dry_run(
        "evaluate-pretrain",
        "DATA_DIR=PDMX",
        "PRETRAIN_CHECKPOINT=artifacts/checkpoints/pretraining/best.pt",
        "EVALUATE_GENERATION_CONFIG=musak_model/configs/evaluation/generation.yml",
        "EVALUATE_SEED=123",
        "EVALUATE_TEMPERATURE=0.8",
    )

    assert "scripts/evaluate_model.py" in output
    assert '"pretrain"' in output
    assert '--data-dir "PDMX"' in output
    assert '--checkpoint "artifacts/checkpoints/pretraining/best.pt"' in output
    assert '--generation-evaluation-config "musak_model/configs/evaluation/generation.yml"' in output
    assert '--seed "123"' in output
    assert '--temperature "0.8"' in output
    assert "--bar-count" not in output
    assert "--max-new-tokens" not in output
    assert "--top-k" not in output


def test_make_evaluate_finetune_dispatches_shared_evaluator() -> None:
    output = _make_dry_run(
        "evaluate-finetune",
        "DATA_DIR=exercises",
        "FINETUNE_CHECKPOINT=artifacts/checkpoints/finetuning/best.pt",
    )

    assert "scripts/evaluate_model.py" in output
    assert '"finetune"' in output
    assert '--data-dir "exercises"' in output
    assert '--checkpoint "artifacts/checkpoints/finetuning/best.pt"' in output
    assert "--bar-count" not in output
    assert "--max-new-tokens" not in output
    assert "--top-k" not in output


def test_make_mlflow_starts_dashboard_with_configurable_address() -> None:
    output = _make_dry_run(
        "mlflow",
        "MLFLOW_DB=artifacts/mlflow/mlflow.db",
        "MLFLOW_HOST=0.0.0.0",
        "MLFLOW_PORT=5050",
    )

    assert "uv run mlflow ui" in output
    assert 'mkdir -p "artifacts/mlflow/"' in output
    assert '--backend-store-uri "sqlite:///artifacts/mlflow/mlflow.db"' in output
    assert '--host "0.0.0.0"' in output
    assert '--port "5050"' in output


def test_make_notebook_target_runs_discovered_marimo_notebook() -> None:
    output = _make_dry_run("notebook-tokenizer-explorer")

    assert 'uv run marimo "edit" "notebooks/tokenizer_explorer.py"' in output


def test_make_notebook_target_supports_marimo_mode_override() -> None:
    output = _make_dry_run("notebook-tokenizer-explorer", "NOTEBOOK_MODE=run")

    assert 'uv run marimo "run" "notebooks/tokenizer_explorer.py"' in output
