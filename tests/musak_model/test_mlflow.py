from pathlib import Path

import pytest

from musak_model.mlflow import (
    flatten_params,
    read_mlflow_run_id,
    resolve_tracking_uri,
    sqlite_tracking_uri,
    write_mlflow_run_id,
)


def test_resolve_tracking_uri_prefers_configured_uri(tmp_path: Path) -> None:
    tracking_uri = resolve_tracking_uri(configured_uri="file:///configured/mlruns", tracking_root=tmp_path)

    assert tracking_uri == "file:///configured/mlruns"


def test_resolve_tracking_uri_uses_environment_before_local_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///env/mlruns")

    tracking_uri = resolve_tracking_uri(configured_uri=None, tracking_root=tmp_path)

    assert tracking_uri == "file:///env/mlruns"


def test_resolve_tracking_uri_uses_local_sqlite_fallback(tmp_path: Path) -> None:
    tracking_uri = resolve_tracking_uri(configured_uri=None, tracking_root=tmp_path)

    assert tracking_uri == f"sqlite:///{tmp_path / 'artifacts/mlflow/mlflow.db'}"
    assert (tmp_path / "artifacts/mlflow").exists()


def test_sqlite_tracking_uri_creates_parent_directory(tmp_path: Path) -> None:
    database_path = tmp_path / "tracking" / "mlflow.db"

    tracking_uri = sqlite_tracking_uri(database_path)

    assert tracking_uri == f"sqlite:///{database_path}"
    assert database_path.parent.exists()


def test_flatten_params_preserves_mlflow_param_values() -> None:
    params = flatten_params(
        {
            "training": {
                "epochs": 3,
                "resume_checkpoint": None,
                "enabled": True,
            },
            "notes": ["a", "b"],
        }
    )

    assert params == {
        "training.epochs": 3,
        "training.resume_checkpoint": "null",
        "training.enabled": True,
        "notes": "['a', 'b']",
    }


def test_mlflow_run_id_sidecar_round_trip(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "checkpoints"

    write_mlflow_run_id(checkpoint_directory=checkpoint_directory, run_id="abc123")

    assert read_mlflow_run_id(checkpoint_directory) == "abc123"
    assert (checkpoint_directory / "mlflow_run_id.txt").read_text(encoding="utf-8") == "abc123\n"


def test_read_mlflow_run_id_returns_none_when_missing_or_empty(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "checkpoints"

    assert read_mlflow_run_id(checkpoint_directory) is None

    checkpoint_directory.mkdir()
    (checkpoint_directory / "mlflow_run_id.txt").write_text("\n", encoding="utf-8")

    assert read_mlflow_run_id(checkpoint_directory) is None
