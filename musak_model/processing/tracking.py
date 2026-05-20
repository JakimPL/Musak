import importlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self

from musak_model.paths import DEFAULT_MLFLOW_DIR
from musak_model.processing.dataset import ProcessDatasetResult
from musak_model.processing.fingerprint import encoded_samples_fingerprint, file_sha256
from musak_model.processing.io import load_encoded_jsonl
from musak_model.processing.manifest import (
    EncodedManifestField,
    ParsedManifestField,
    ParsedManifestStatus,
    read_encoded_manifest,
    read_parsed_manifest,
)
from musak_model.processing.paths import ENCODED_JSONL_NAME

_MLFLOW_TRACKING_URI_ENV: Final[str] = "MLFLOW_TRACKING_URI"
_TRUE_TEXT: Final[str] = "True"


class _MlflowLogger(Protocol):
    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None: ...


@dataclass(frozen=True)
class ProcessingMlflowConfig:
    enabled: bool = True
    experiment_name: str = "musak-process"
    run_name: str | None = None
    tracking_uri: str | None = None


class ProcessingTracker:
    def __init__(self, *, config: ProcessingMlflowConfig, tracking_root: Path | None = None) -> None:
        self._config = config
        self._tracking_root = tracking_root or Path.cwd()
        self._mlflow = importlib.import_module("mlflow")

    def __enter__(self) -> Self:
        self._mlflow.set_tracking_uri(
            _resolve_tracking_uri(configured_uri=self._config.tracking_uri, tracking_root=self._tracking_root)
        )
        self._mlflow.set_experiment(self._config.experiment_name)
        self._mlflow.start_run(run_name=self._config.run_name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self._mlflow.end_run(status=status)

    def log_processing_result(
        self,
        *,
        result: ProcessDatasetResult,
        data_dir: Path,
        processed_root: Path,
        stage: str,
        overwrite: bool,
    ) -> None:
        params = _processing_params(
            result=result,
            data_dir=data_dir,
            processed_root=processed_root,
            stage=stage,
            overwrite=overwrite,
        )
        metrics = _processing_metrics(result=result)
        self._mlflow.log_params(params)
        for name, value in metrics.items():
            if math.isfinite(value):
                self._mlflow.log_metric(name, value)

        _log_artifact_if_exists(self._mlflow, result.parsed_manifest_path, artifact_path="dataset")
        _log_artifact_if_exists(self._mlflow, result.encoded_manifest_path, artifact_path="dataset")
        _log_artifact_if_exists(self._mlflow, result.tokenizer_snapshot_path, artifact_path="dataset")


class NoOpProcessingTracker:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def log_processing_result(
        self,
        *,
        result: ProcessDatasetResult,
        data_dir: Path,
        processed_root: Path,
        stage: str,
        overwrite: bool,
    ) -> None:
        return None


def build_processing_tracker(
    *,
    config: ProcessingMlflowConfig,
    tracking_root: Path | None = None,
) -> ProcessingTracker | NoOpProcessingTracker:
    if not config.enabled:
        return NoOpProcessingTracker()

    return ProcessingTracker(config=config, tracking_root=tracking_root)


def _processing_params(
    *,
    result: ProcessDatasetResult,
    data_dir: Path,
    processed_root: Path,
    stage: str,
    overwrite: bool,
) -> dict[str, str | int | float | bool]:
    params: dict[str, str | int | float | bool] = {
        "data.dataset_name": data_dir.name,
        "data.data_dir": str(data_dir),
        "data.processed_root": str(processed_root),
        "data.processed_artifact_dir": str(result.parsed_manifest_path.parent),
        "processing.stage": stage,
        "processing.overwrite": overwrite,
        "data.parsed_manifest_sha256": file_sha256(result.parsed_manifest_path),
    }
    if result.encoded_manifest_path is not None and result.encoded_manifest_path.exists():
        params["data.encoded_manifest_sha256"] = file_sha256(result.encoded_manifest_path)

    if result.tokenizer_snapshot_path is not None and result.tokenizer_snapshot_path.exists():
        params["data.tokenizer_hash"] = result.tokenizer_snapshot_path.parent.name
        params["data.tokenizer_snapshot_sha256"] = file_sha256(result.tokenizer_snapshot_path)

    encoded_jsonl_path = _encoded_jsonl_path(result)
    if encoded_jsonl_path is not None and encoded_jsonl_path.exists():
        params["data.encoded_samples_fingerprint"] = encoded_samples_fingerprint(load_encoded_jsonl(encoded_jsonl_path))
        params["data.encoded_jsonl_sha256"] = file_sha256(encoded_jsonl_path)

    return params


def _processing_metrics(*, result: ProcessDatasetResult) -> dict[str, float]:
    parsed_rows = read_parsed_manifest(result.parsed_manifest_path)
    parsed_count = len(parsed_rows)
    parsed_success_count = sum(
        row[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value for row in parsed_rows
    )
    metrics = {
        "dataset/overall/count/parsed_files": float(parsed_count),
        "dataset/overall/count/parsed_successes": float(parsed_success_count),
        "dataset/overall/count/parse_errors": float(parsed_count - parsed_success_count),
        "dataset/overall/rate/parse_success": _rate(parsed_success_count, parsed_count),
    }
    if result.encoded_manifest_path is None or not result.encoded_manifest_path.exists():
        return metrics

    encoded_rows = read_encoded_manifest(result.encoded_manifest_path)
    segment_count = len(encoded_rows)
    encoded_count = sum(row[EncodedManifestField.ENCODED_LINE] != "" for row in encoded_rows)
    eligible_count = sum(row[EncodedManifestField.ELIGIBLE_FOR_TRAINING] == _TRUE_TEXT for row in encoded_rows)
    metrics.update(
        {
            "dataset/overall/count/segments": float(segment_count),
            "dataset/overall/count/encoded_samples": float(encoded_count),
            "dataset/overall/rate/eligible": _rate(eligible_count, segment_count),
            "dataset/diagnostics/rate/empty_score": _boolean_rate(encoded_rows, EncodedManifestField.EMPTY_SCORE),
            "dataset/diagnostics/rate/one_hand_only": _boolean_rate(encoded_rows, EncodedManifestField.ONE_HAND_ONLY),
            "dataset/diagnostics/mean/right_silence_fraction": _numeric_mean(
                encoded_rows,
                EncodedManifestField.RIGHT_SILENCE_FRACTION,
            ),
            "dataset/diagnostics/mean/left_silence_fraction": _numeric_mean(
                encoded_rows,
                EncodedManifestField.LEFT_SILENCE_FRACTION,
            ),
            "dataset/diagnostics/mean/both_hands_silence_fraction": _numeric_mean(
                encoded_rows,
                EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION,
            ),
            "dataset/diagnostics/mean/both_hands_active_fraction": _numeric_mean(
                encoded_rows,
                EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION,
            ),
            "dataset/diagnostics/mean/hand_activity_balance": _numeric_mean(
                encoded_rows,
                EncodedManifestField.HAND_ACTIVITY_BALANCE,
            ),
            "dataset/diagnostics/mean/silent_bar_count": _numeric_mean(
                encoded_rows,
                EncodedManifestField.SILENT_BAR_COUNT,
            ),
            "dataset/diagnostics/mean/silent_bar_fraction": _numeric_mean(
                encoded_rows,
                EncodedManifestField.SILENT_BAR_FRACTION,
            ),
            "dataset/diagnostics/mean/silent_edge_bar_count": _numeric_mean(
                encoded_rows,
                EncodedManifestField.SILENT_EDGE_BAR_COUNT,
            ),
            "dataset/tokens/mean/note_fraction": _numeric_mean(
                encoded_rows,
                EncodedManifestField.NOTE_TOKEN_FRACTION,
            ),
            "dataset/tokens/mean/rest_fraction": _numeric_mean(
                encoded_rows,
                EncodedManifestField.REST_TOKEN_FRACTION,
            ),
            "dataset/tokens/mean/hold_fraction": _numeric_mean(
                encoded_rows,
                EncodedManifestField.HOLD_TOKEN_FRACTION,
            ),
        }
    )
    for reason, count in _ineligibility_reason_counts(encoded_rows).items():
        metrics[f"dataset/ineligibility/count/{reason}"] = float(count)
        metrics[f"dataset/ineligibility/rate/{reason}"] = _rate(count, segment_count)

    return metrics


def _encoded_jsonl_path(result: ProcessDatasetResult) -> Path | None:
    if result.encoded_manifest_path is None:
        return None

    return result.encoded_manifest_path.parent / ENCODED_JSONL_NAME


def _numeric_mean(rows: list[dict[str, str]], field: EncodedManifestField) -> float:
    values = [float(row[field]) for row in rows if row[field] != ""]
    if not values:
        return math.nan

    return sum(values) / len(values)


def _ineligibility_reason_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason_text = row[EncodedManifestField.INELIGIBILITY_REASONS]
        if reason_text == "":
            continue

        for reason in reason_text.split("|"):
            counts[reason] = counts.get(reason, 0) + 1

    return counts


def _boolean_rate(rows: list[dict[str, str]], field: EncodedManifestField) -> float:
    if not rows:
        return math.nan

    return sum(row[field] == _TRUE_TEXT for row in rows) / len(rows)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan

    return numerator / denominator


def _resolve_tracking_uri(*, configured_uri: str | None, tracking_root: Path) -> str:
    if configured_uri is not None:
        return configured_uri

    environment_uri = os.getenv(_MLFLOW_TRACKING_URI_ENV)
    if environment_uri:
        return environment_uri

    return str(tracking_root / DEFAULT_MLFLOW_DIR)


def _log_artifact_if_exists(mlflow: _MlflowLogger, path: Path | None, *, artifact_path: str) -> None:
    if path is not None and path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
