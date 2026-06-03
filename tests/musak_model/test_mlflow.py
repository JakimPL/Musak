import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from musak_model.mlflow import (
    MlflowRun,
    MlflowRunConfig,
    flatten_params,
    read_mlflow_run_id,
    resolve_tracking_uri,
    sqlite_tracking_uri,
    write_mlflow_run_id,
)


class FakeRunInfo:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class FakeActiveRun:
    def __init__(self, run_id: str) -> None:
        self.info = FakeRunInfo(run_id)


class FakeMlflow(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.run_name: str | None = None
        self.run_id: str | None = None
        self.ended_status: str | None = None
        self.tags: dict[str, str] = {}
        self.params: dict[str, str | int | float | bool] = {}
        self.metrics: list[tuple[str, float, int | None]] = []
        self.logged_dicts: list[tuple[dict[str, object], str]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.artifact_directories: list[tuple[str, str | None]] = []

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def start_run(self, *, run_id: str | None = None, run_name: str | None = None) -> FakeActiveRun:
        self.run_id = run_id or "generated-run-id"
        self.run_name = run_name
        return FakeActiveRun(self.run_id)

    def end_run(self, *, status: str) -> None:
        self.ended_status = status

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        self.metrics.append((key, value, step))

    def log_dict(self, dictionary: dict[str, object], artifact_file: str) -> None:
        self.logged_dicts.append((dictionary, artifact_file))

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))

    def log_artifacts(self, local_dir: str, *, artifact_path: str | None = None) -> None:
        self.artifact_directories.append((local_dir, artifact_path))


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


def test_mlflow_run_lifecycle_and_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")
    artifact_directory = tmp_path / "artifact-directory"
    artifact_directory.mkdir()

    with MlflowRun(
        MlflowRunConfig(
            enabled=True,
            experiment_name="experiment",
            run_name="run-name",
            run_id=None,
            tracking_uri="file:///tmp/mlruns",
        ),
        tracking_root=tmp_path,
    ) as run:
        assert run.enabled
        assert run.run_id == "generated-run-id"
        run.set_tag("kind", "test")
        run.log_params({"alpha": 1, "enabled": True})
        run.log_metric("finite", 0.5, step=3)
        run.log_metric("nan", math.nan, step=3)
        run.log_dict({"value": "payload"}, "payload.json")
        run.log_artifact(artifact, artifact_path="single")
        run.log_artifacts(artifact_directory, artifact_path="directory")

    assert fake_mlflow.tracking_uri == "file:///tmp/mlruns"
    assert fake_mlflow.experiment_name == "experiment"
    assert fake_mlflow.run_name == "run-name"
    assert fake_mlflow.ended_status == "FINISHED"
    assert fake_mlflow.tags == {"kind": "test"}
    assert fake_mlflow.params == {"alpha": 1, "enabled": True}
    assert fake_mlflow.metrics == [("finite", 0.5, 3)]
    assert fake_mlflow.logged_dicts == [({"value": "payload"}, "payload.json")]
    assert fake_mlflow.artifacts == [(str(artifact), "single")]
    assert fake_mlflow.artifact_directories == [(str(artifact_directory), "directory")]


def test_mlflow_run_attaches_by_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with MlflowRun(
        MlflowRunConfig(
            enabled=True,
            experiment_name="experiment",
            run_name="ignored-name",
            run_id="abc123",
            tracking_uri="file:///tmp/mlruns",
        ),
        tracking_root=tmp_path,
    ) as run:
        assert run.run_id == "abc123"

    assert fake_mlflow.run_id == "abc123"
    assert fake_mlflow.run_name is None


def test_disabled_mlflow_run_is_noop(tmp_path: Path) -> None:
    with MlflowRun(
        MlflowRunConfig(
            enabled=False,
            experiment_name="experiment",
            run_name="run-name",
            run_id=None,
            tracking_uri="file:///tmp/mlruns",
        ),
        tracking_root=tmp_path,
    ) as run:
        assert not run.enabled
        assert run.run_id is None
        run.set_tag("kind", "test")
        run.log_params({"alpha": 1})
        run.log_metric("finite", 0.5, step=3)
        run.log_metrics({"other": 1.0}, step=3)
        run.log_dict({"value": "payload"}, "payload.json")
        run.log_artifact(tmp_path / "missing.txt")
        run.log_artifacts(tmp_path / "missing-directory")


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
