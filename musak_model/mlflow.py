from __future__ import annotations

import importlib
import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Final, Self

from pydantic import BaseModel, ConfigDict

from musak_model.paths import DEFAULT_MLFLOW_DB_PATH, ROOT_DIRECTORY

_LOGGER = logging.getLogger(__name__)
_MLFLOW_TRACKING_URI_ENV: Final[str] = "MLFLOW_TRACKING_URI"

type ParamValue = str | int | float | bool


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

    return local_mlflow_tracking_uri(database_path=DEFAULT_MLFLOW_DB_PATH, tracking_root=tracking_root)


def serializable_dump(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="json")


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


class MlflowRunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    experiment_name: str = "musak"
    run_name: str | None = None
    tracking_uri: str | None = None


class MlflowRun:
    def __init__(self, config: MlflowRunConfig, *, tracking_root: Path | None = None) -> None:
        self._config = config
        self._tracking_root = tracking_root or Path.cwd()
        self._mlflow = importlib.import_module("mlflow") if config.enabled else None

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    def __enter__(self) -> Self:
        if self._mlflow is None:
            return self

        tracking_uri = resolve_tracking_uri(configured_uri=self._config.tracking_uri, tracking_root=self._tracking_root)
        _LOGGER.info(
            "Starting MLflow run: experiment=%s run_name=%s tracking_uri=%s",
            self._config.experiment_name,
            self._config.run_name,
            tracking_uri,
        )
        self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(self._config.experiment_name)
        self._mlflow.start_run(run_name=self._config.run_name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(dict(params))

    def log_metric(self, name: str, value: float, *, step: int | None = None) -> None:
        if self._mlflow is not None and math.isfinite(value):
            self._mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_dict(self, payload: dict[str, object], artifact_file: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_dict(payload, artifact_file)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        if self._mlflow is not None and path.exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
