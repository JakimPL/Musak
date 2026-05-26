from __future__ import annotations

import importlib
import logging
import math
import os
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Final, Protocol, Self

from pydantic import BaseModel

from musak_model.model.config import ModelConfig
from musak_model.paths import DEFAULT_MLFLOW_DIR
from musak_model.processing.fingerprint import encoded_samples_fingerprint
from musak_model.training.config import TrainingConfig
from musak_model.training.ingestion.schema import IngestionErrorRecord, IngestionSplit
from musak_model.training.metrics import EpochMetrics

_LOGGER = logging.getLogger(__name__)
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

    def log_generation_evaluation(self, *, metrics: dict[str, float], epoch: int) -> None: ...

    def log_split_figure_metrics(self, *, metrics: dict[str, float]) -> None: ...

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

    def log_generation_evaluation(self, *, metrics: dict[str, float], epoch: int) -> None:
        return None

    def log_split_figure_metrics(self, *, metrics: dict[str, float]) -> None:
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
        _LOGGER.info(
            "Starting MLflow training run: experiment=%s run_name=%s tracking_uri=%s",
            self._training_config.mlflow.mlflow_experiment_name,
            self._training_config.mlflow.mlflow_run_name,
            tracking_uri,
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
        _LOGGER.info(
            "Logging MLflow training setup: train_samples=%s validation_samples=%s invalid_files=%s",
            len(split.train),
            len(split.validation),
            len(split.invalid_files),
        )
        started_at = perf_counter()
        _LOGGER.info("Computing encoded sample fingerprint for MLflow setup")
        fingerprint = encoded_samples_fingerprint([*split.train, *split.validation])
        params = _flatten_params(
            {
                "training": _serializable_dump(training_config),
                "model": _serializable_dump(model_config),
                "data": {
                    "train_samples": len(split.train),
                    "validation_samples": len(split.validation),
                    "invalid_files": len(split.invalid_files),
                    "encoded_samples_fingerprint": fingerprint,
                },
            }
        )
        self._mlflow.log_params(params)
        _LOGGER.info("Logged MLflow training setup in %.1fs", perf_counter() - started_at)

    def log_epoch(
        self,
        *,
        metrics: EpochMetrics,
    ) -> None:
        for name, value in _epoch_metric_values(metrics).items():
            self._mlflow.log_metric(name, value, step=metrics.epoch)

    def log_generation_evaluation(self, *, metrics: dict[str, float], epoch: int) -> None:
        _LOGGER.info("Logging %s generation evaluation metric(s) to MLflow", len(metrics))
        for name, value in metrics.items():
            if math.isfinite(value):
                self._mlflow.log_metric(name, value, step=epoch)

    def log_split_figure_metrics(self, *, metrics: dict[str, float]) -> None:
        _LOGGER.info("Logging %s split figure metric(s) to MLflow", len(metrics))
        for name, value in metrics.items():
            if math.isfinite(value):
                self._mlflow.log_metric(name, value, step=0)

    def log_checkpoints(
        self,
        *,
        latest_checkpoint_path: Path | None,
        best_checkpoint_path: Path | None,
    ) -> None:
        if latest_checkpoint_path is not None and latest_checkpoint_path.exists():
            _LOGGER.info("Logging latest checkpoint artifact to MLflow: %s", latest_checkpoint_path)
            self._mlflow.log_artifact(str(latest_checkpoint_path), artifact_path="checkpoints")

        if best_checkpoint_path is not None and best_checkpoint_path.exists():
            _LOGGER.info("Logging best checkpoint artifact to MLflow: %s", best_checkpoint_path)
            self._mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")

    def log_invalid_files(self, *, invalid_files: list[IngestionErrorRecord]) -> None:
        if invalid_files:
            _LOGGER.info("Logging invalid file report to MLflow: invalid_files=%s", len(invalid_files))
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
    return {
        metric_name: value
        for field_name, metric_name in _EPOCH_METRIC_NAME_MAP.items()
        if isinstance((value := getattr(metrics, field_name)), float)
    }


_EPOCH_METRIC_NAME_MAP: Final[dict[str, str]] = {
    "train_loss": "model/train/mean/loss",
    "train_perplexity": "model/train/mean/perplexity",
    "train_token_accuracy": "model/train/rate/token_accuracy",
    "train_token_kind_accuracy": "model/train/rate/token_kind_accuracy",
    "train_validity_penalty_loss": "model/train/mean/validity_penalty_loss",
    "train_invalid_probability_mass": "model/train/mean/invalid_probability_mass",
    "train_invalid_target_rate": "model/train/rate/invalid_target",
    "train_cnn_gradient_norm": "model/train/mean/cnn_gradient_norm",
    "train_gru_gradient_norm": "model/train/mean/gru_gradient_norm",
    "train_transformer_gradient_norm": "model/train/mean/transformer_gradient_norm",
    "validation_loss": "model/validation/mean/loss",
    "validation_perplexity": "model/validation/mean/perplexity",
    "validation_token_accuracy": "model/validation/rate/token_accuracy",
    "validation_token_kind_accuracy": "model/validation/rate/token_kind_accuracy",
    "validation_validity_penalty_loss": "model/validation/mean/validity_penalty_loss",
    "validation_invalid_probability_mass": "model/validation/mean/invalid_probability_mass",
    "validation_invalid_target_rate": "model/validation/rate/invalid_target",
}


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
