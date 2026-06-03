from __future__ import annotations

import importlib
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self, cast

from musak_model.paths import DEFAULT_MLFLOW_DB_PATH, ROOT_DIRECTORY

_LOGGER = logging.getLogger(__name__)
_MLFLOW_TRACKING_URI_ENV: Final[str] = "MLFLOW_TRACKING_URI"
MLFLOW_RUN_ID_FILE_NAME: Final[str] = "mlflow_run_id.txt"

type ParamValue = str | int | float | bool


class _MlflowRunInfo(Protocol):
    run_id: str


class _ActiveMlflowRun(Protocol):
    info: _MlflowRunInfo


class _MlflowModule(Protocol):
    def set_tracking_uri(self, tracking_uri: str) -> None: ...

    def set_experiment(self, experiment_name: str) -> None: ...

    def start_run(
        self,
        *,
        run_id: str | None = None,
        run_name: str | None = None,
    ) -> _ActiveMlflowRun: ...

    def end_run(self, *, status: str) -> None: ...

    def set_tag(self, key: str, value: str) -> None: ...

    def log_params(self, params: dict[str, ParamValue]) -> None: ...

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None: ...

    def log_dict(self, dictionary: dict[str, object], artifact_file: str) -> None: ...

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None: ...

    def log_artifacts(self, local_dir: str, *, artifact_path: str | None = None) -> None: ...


@dataclass(frozen=True)
class MlflowRunConfig:
    enabled: bool
    experiment_name: str
    run_name: str | None
    run_id: str | None
    tracking_uri: str | None


class MlflowRun:
    def __init__(self, config: MlflowRunConfig, *, tracking_root: Path | None = None) -> None:
        self._config = config
        self._tracking_root = tracking_root or Path.cwd()
        self._mlflow = cast(_MlflowModule, importlib.import_module("mlflow")) if config.enabled else None
        self._run_id: str | None = config.run_id

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def __enter__(self) -> Self:
        if self._mlflow is None:
            return self

        tracking_uri = resolve_tracking_uri(
            configured_uri=self._config.tracking_uri,
            tracking_root=self._tracking_root,
        )
        _LOGGER.info(
            "Starting MLflow run: experiment=%s run_name=%s run_id=%s tracking_uri=%s",
            self._config.experiment_name,
            self._config.run_name,
            self._config.run_id,
            tracking_uri,
        )
        self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(self._config.experiment_name)
        active_run = self._mlflow.start_run(
            run_id=self._config.run_id,
            run_name=None if self._config.run_id is not None else self._config.run_name,
        )
        self._run_id = active_run.info.run_id
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._mlflow is None:
            return

        self._mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")

    def set_tag(self, key: str, value: str) -> None:
        if self._mlflow is not None:
            self._mlflow.set_tag(key, value)

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(dict(params))

    def log_metric(self, name: str, value: float, *, step: int | None = None) -> None:
        if self._mlflow is None or not math.isfinite(value):
            return

        if step is None:
            self._mlflow.log_metric(name, value)
        else:
            self._mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_dict(self, payload: dict[str, object], artifact_file: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_dict(payload, artifact_file)

    def log_artifact(self, path: Path | None, *, artifact_path: str | None = None) -> None:
        if self._mlflow is not None and path is not None and path.exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, directory: Path, *, artifact_path: str | None = None) -> None:
        if self._mlflow is not None and directory.exists():
            self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)


def sqlite_tracking_uri(database_path: Path) -> str:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path.as_posix()}"


def local_mlflow_tracking_uri(
    *,
    database_path: Path,
    tracking_root: Path,
) -> str:
    if database_path.is_absolute():
        try:
            database_path = database_path.relative_to(ROOT_DIRECTORY)
        except ValueError:
            return sqlite_tracking_uri(database_path)

    return sqlite_tracking_uri(tracking_root / database_path)


def resolve_tracking_uri(*, configured_uri: str | None, tracking_root: Path) -> str:
    if configured_uri is not None:
        return configured_uri

    environment_uri = os.getenv(_MLFLOW_TRACKING_URI_ENV)
    if environment_uri:
        return environment_uri

    return local_mlflow_tracking_uri(
        database_path=DEFAULT_MLFLOW_DB_PATH,
        tracking_root=tracking_root,
    )


def mlflow_run_id_path(checkpoint_directory: Path) -> Path:
    return checkpoint_directory / MLFLOW_RUN_ID_FILE_NAME


def write_mlflow_run_id(*, checkpoint_directory: Path, run_id: str) -> None:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    mlflow_run_id_path(checkpoint_directory).write_text(f"{run_id}\n", encoding="utf-8")


def read_mlflow_run_id(checkpoint_directory: Path) -> str | None:
    run_id_path = mlflow_run_id_path(checkpoint_directory)
    if not run_id_path.exists():
        return None

    run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        return None

    return run_id


def flatten_params(values: dict[str, object], *, prefix: str = "") -> dict[str, ParamValue]:
    flattened: dict[str, ParamValue] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(flatten_params(value, prefix=f"{name}."))
        elif isinstance(value, (str, int, float, bool)):
            flattened[name] = value
        elif value is None:
            flattened[name] = "null"
        else:
            flattened[name] = str(value)

    return flattened
