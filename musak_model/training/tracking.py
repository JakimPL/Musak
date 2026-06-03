from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Final, Protocol, Self

from musak_model.mlflow import MlflowRun, flatten_params, serializable_dump
from musak_model.model.config import ModelConfig
from musak_model.processing.fingerprint import encoded_samples_fingerprint
from musak_model.training.config import TrainingConfig
from musak_model.training.ingestion.schema import IngestionErrorRecord, IngestionSplit
from musak_model.training.metrics import EpochMetrics

_LOGGER = logging.getLogger(__name__)


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
        self._run = MlflowRun(training_config.mlflow, tracking_root=tracking_root)

    def __enter__(self) -> Self:
        self._run.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._run.__exit__(exc_type, exc_value, traceback)

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
        self._run.log_params(
            flatten_params(
                {
                    "training": serializable_dump(training_config),
                    "model": serializable_dump(model_config),
                    "data": {
                        "train_samples": len(split.train),
                        "validation_samples": len(split.validation),
                        "invalid_files": len(split.invalid_files),
                        "encoded_samples_fingerprint": fingerprint,
                    },
                }
            )
        )
        _LOGGER.info("Logged MLflow training setup in %.1fs", perf_counter() - started_at)

    def log_epoch(
        self,
        *,
        metrics: EpochMetrics,
    ) -> None:
        self._run.log_metrics(_epoch_metric_values(metrics), step=metrics.epoch)

    def log_generation_evaluation(self, *, metrics: dict[str, float], epoch: int) -> None:
        _LOGGER.info("Logging %s generation evaluation metric(s) to MLflow", len(metrics))
        self._run.log_metrics(metrics, step=epoch)

    def log_split_figure_metrics(self, *, metrics: dict[str, float]) -> None:
        _LOGGER.info("Logging %s split figure metric(s) to MLflow", len(metrics))
        self._run.log_metrics(metrics, step=0)

    def log_checkpoints(
        self,
        *,
        latest_checkpoint_path: Path | None,
        best_checkpoint_path: Path | None,
    ) -> None:
        for checkpoint_path in (latest_checkpoint_path, best_checkpoint_path):
            if checkpoint_path is not None:
                self._run.log_artifact(checkpoint_path, artifact_path="checkpoints")

    def log_invalid_files(self, *, invalid_files: list[IngestionErrorRecord]) -> None:
        if invalid_files:
            _LOGGER.info("Logging invalid file report to MLflow: invalid_files=%s", len(invalid_files))
            self._run.log_dict(
                {"invalid_files": [serializable_dump(record) for record in invalid_files]},
                "invalid_files.json",
            )


def build_training_tracker(
    *,
    training_config: TrainingConfig,
    tracking_root: Path | None = None,
) -> TrainingTracker:
    if not training_config.mlflow.enabled:
        return NoOpTrainingTracker()

    return MlflowTrainingTracker(training_config=training_config, tracking_root=tracking_root)


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
