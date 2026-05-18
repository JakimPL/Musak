from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self

from pydantic import BaseModel

from musak_model.model.config import ModelConfig
from musak_model.paths import DEFAULT_MLFLOW_DIR
from musak_model.training.config import TrainingConfig
from musak_model.training.ingestion.schema import IngestionErrorRecord, IngestionSplit
from musak_model.training.metrics import EpochMetrics

_MLFLOW_TRACKING_URI_ENV: Final[str] = "MLFLOW_TRACKING_URI"


class TrainingTracker(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def log_setup(
        self,
        *,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        split: IngestionSplit,
    ) -> None: ...

    def log_epoch(self, *, metrics: EpochMetrics) -> None: ...

    def log_checkpoints(self, *, latest_checkpoint_path: Path | None, best_checkpoint_path: Path | None) -> None: ...

    def log_invalid_files(self, *, invalid_files: list[IngestionErrorRecord]) -> None: ...


class NoOpTrainingTracker:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def log_setup(
        self,
        *,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        split: IngestionSplit,
    ) -> None:
        return None

    def log_epoch(
        self,
        *,
        metrics: EpochMetrics,
    ) -> None:
        return None

    def log_checkpoints(
        self,
        *,
        latest_checkpoint_path: Path | None,
        best_checkpoint_path: Path | None,
    ) -> None:
        return None

    def log_invalid_files(
        self,
        *,
        invalid_files: list[IngestionErrorRecord],
    ) -> None:
        return None


class MlflowTrainingTracker:
    def __init__(
        self,
        *,
        training_config: TrainingConfig,
        tracking_root: Path | None = None,
    ) -> None:
        self._training_config = training_config
        self._tracking_root = tracking_root or Path.cwd()
        self._mlflow = importlib.import_module("mlflow")

    def __enter__(self) -> Self:
        tracking_uri = _resolve_tracking_uri(
            configured_uri=self._training_config.mlflow.mlflow_tracking_uri,
            tracking_root=self._tracking_root,
        )
        self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(self._training_config.mlflow.mlflow_experiment_name)
        self._mlflow.start_run(run_name=self._training_config.mlflow.mlflow_run_name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self._mlflow.end_run(status=status)

    def log_setup(
        self,
        *,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        split: IngestionSplit,
    ) -> None:
        params = _flatten_params(
            {
                "training": _serializable_dump(training_config),
                "model": _serializable_dump(model_config),
                "data": {
                    "train_samples": len(split.train),
                    "validation_samples": len(split.validation),
                    "invalid_files": len(split.invalid_files),
                },
            }
        )
        self._mlflow.log_params(params)

    def log_epoch(
        self,
        *,
        metrics: EpochMetrics,
    ) -> None:
        for name, value in _epoch_metric_values(metrics).items():
            self._mlflow.log_metric(name, value, step=metrics.epoch)

    def log_checkpoints(
        self,
        *,
        latest_checkpoint_path: Path | None,
        best_checkpoint_path: Path | None,
    ) -> None:
        if latest_checkpoint_path is not None and latest_checkpoint_path.exists():
            self._mlflow.log_artifact(str(latest_checkpoint_path), artifact_path="checkpoints")

        if best_checkpoint_path is not None and best_checkpoint_path.exists():
            self._mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")

    def log_invalid_files(self, *, invalid_files: list[IngestionErrorRecord]) -> None:
        if invalid_files:
            self._mlflow.log_dict(
                {"invalid_files": [_serializable_dump(record) for record in invalid_files]},
                "invalid_files.json",
            )


def build_training_tracker(
    *,
    training_config: TrainingConfig,
    tracking_root: Path | None = None,
) -> TrainingTracker:
    if not training_config.mlflow.enable_mlflow:
        return NoOpTrainingTracker()

    return MlflowTrainingTracker(training_config=training_config, tracking_root=tracking_root)


def _resolve_tracking_uri(
    *,
    configured_uri: str | None,
    tracking_root: Path,
) -> str:
    if configured_uri is not None:
        return configured_uri

    environment_uri = os.getenv(_MLFLOW_TRACKING_URI_ENV)
    if environment_uri:
        return environment_uri

    return str(tracking_root / DEFAULT_MLFLOW_DIR)


def _serializable_dump(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="json")


def _epoch_metric_values(metrics: EpochMetrics) -> dict[str, float]:
    return {key: value for key, value in metrics.model_dump().items() if key != "epoch" and isinstance(value, float)}


def _flatten_params(values: dict[str, object]) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    _flatten_into(flattened=flattened, prefix="", value=values)
    return flattened


def _flatten_into(
    *,
    flattened: dict[str, str | int | float | bool],
    prefix: str,
    value: object,
) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_into(flattened=flattened, prefix=nested_prefix, value=nested_value)
        return

    if isinstance(value, (str, int, float, bool)):
        flattened[prefix] = value
        return

    if value is None:
        flattened[prefix] = "null"
        return

    flattened[prefix] = str(value)
