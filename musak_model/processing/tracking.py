import importlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self

from musak_model.paths import DEFAULT_MLFLOW_DIR
from musak_model.processing.dataset import ProcessDatasetResult
from musak_model.processing.fingerprint import encoded_samples_jsonl_fingerprint, file_sha256
from musak_model.processing.manifest import (
    EncodedManifestField,
    ParsedManifestField,
    ParsedManifestStatus,
    iter_encoded_manifest,
    iter_parsed_manifest,
)
from musak_model.processing.paths import ENCODED_JSONL_NAME

_MLFLOW_TRACKING_URI_ENV: Final[str] = "MLFLOW_TRACKING_URI"
_TRUE_TEXT: Final[str] = "True"
_DIAGNOSTIC_BOOLEAN_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.EMPTY_SCORE,
    EncodedManifestField.ONE_HAND_ONLY,
    EncodedManifestField.SCALE_MATCH_DECLARED_MATCH_USED,
    EncodedManifestField.SCALE_MATCH_LOW_CONFIDENCE,
    EncodedManifestField.SCALE_MATCH_AMBIGUOUS,
    EncodedManifestField.SCALE_MATCH_NO_PITCHES,
)
_DIAGNOSTIC_NUMERIC_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.RIGHT_SILENCE_FRACTION,
    EncodedManifestField.LEFT_SILENCE_FRACTION,
    EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION,
    EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION,
    EncodedManifestField.HAND_ACTIVITY_BALANCE,
    EncodedManifestField.SILENT_BAR_COUNT,
    EncodedManifestField.SILENT_BAR_FRACTION,
    EncodedManifestField.SILENT_EDGE_BAR_COUNT,
    EncodedManifestField.NOTE_TOKEN_FRACTION,
    EncodedManifestField.REST_TOKEN_FRACTION,
    EncodedManifestField.HOLD_TOKEN_FRACTION,
    EncodedManifestField.SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_EXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_UNEXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_BEST_MARGIN,
    EncodedManifestField.SCALE_MATCH_OBSERVED_PITCH_CLASS_COUNT,
    EncodedManifestField.SCALE_MATCH_EXPLANATION_PITCH_CLASS_COUNT,
    EncodedManifestField.SCALE_MATCH_SUPPORT_CANDIDATE_COUNT,
    EncodedManifestField.SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT,
)
_NUMERIC_METRIC_NAMES: Final[dict[EncodedManifestField, str]] = {
    EncodedManifestField.RIGHT_SILENCE_FRACTION: "dataset/diagnostics/mean/right_silence_fraction",
    EncodedManifestField.LEFT_SILENCE_FRACTION: "dataset/diagnostics/mean/left_silence_fraction",
    EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION: "dataset/diagnostics/mean/both_hands_silence_fraction",
    EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION: "dataset/diagnostics/mean/both_hands_active_fraction",
    EncodedManifestField.HAND_ACTIVITY_BALANCE: "dataset/diagnostics/mean/hand_activity_balance",
    EncodedManifestField.SILENT_BAR_COUNT: "dataset/diagnostics/mean/silent_bar_count",
    EncodedManifestField.SILENT_BAR_FRACTION: "dataset/diagnostics/mean/silent_bar_fraction",
    EncodedManifestField.SILENT_EDGE_BAR_COUNT: "dataset/diagnostics/mean/silent_edge_bar_count",
    EncodedManifestField.NOTE_TOKEN_FRACTION: "dataset/tokens/mean/note_fraction",
    EncodedManifestField.REST_TOKEN_FRACTION: "dataset/tokens/mean/rest_fraction",
    EncodedManifestField.HOLD_TOKEN_FRACTION: "dataset/tokens/mean/hold_fraction",
    EncodedManifestField.SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION: "dataset/scale_match/mean/in_scale_weight_fraction",
    EncodedManifestField.SCALE_MATCH_OUT_OF_SCALE_WEIGHT_FRACTION: (
        "dataset/scale_match/mean/out_of_scale_weight_fraction"
    ),
    EncodedManifestField.SCALE_MATCH_EXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION: (
        "dataset/scale_match/mean/explained_out_of_scale_weight_fraction"
    ),
    EncodedManifestField.SCALE_MATCH_UNEXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION: (
        "dataset/scale_match/mean/unexplained_out_of_scale_weight_fraction"
    ),
    EncodedManifestField.SCALE_MATCH_BEST_MARGIN: "dataset/scale_match/mean/best_margin",
    EncodedManifestField.SCALE_MATCH_OBSERVED_PITCH_CLASS_COUNT: (
        "dataset/scale_match/mean/observed_pitch_class_count"
    ),
    EncodedManifestField.SCALE_MATCH_EXPLANATION_PITCH_CLASS_COUNT: (
        "dataset/scale_match/mean/explanation_pitch_class_count"
    ),
    EncodedManifestField.SCALE_MATCH_SUPPORT_CANDIDATE_COUNT: ("dataset/scale_match/mean/support_candidate_count"),
    EncodedManifestField.SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT: ("dataset/scale_match/mean/tied_best_candidate_count"),
}
_BOOLEAN_METRIC_NAMES: Final[dict[EncodedManifestField, str]] = {
    EncodedManifestField.EMPTY_SCORE: "dataset/diagnostics/rate/empty_score",
    EncodedManifestField.ONE_HAND_ONLY: "dataset/diagnostics/rate/one_hand_only",
    EncodedManifestField.SCALE_MATCH_DECLARED_MATCH_USED: "dataset/scale_match/rate/declared_match_used",
    EncodedManifestField.SCALE_MATCH_LOW_CONFIDENCE: "dataset/scale_match/rate/low_confidence",
    EncodedManifestField.SCALE_MATCH_AMBIGUOUS: "dataset/scale_match/rate/ambiguous",
    EncodedManifestField.SCALE_MATCH_NO_PITCHES: "dataset/scale_match/rate/no_pitches",
}


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
        "processing.scale_match_support_score_margin": result.scale_matcher_config.support_score_margin,
        "processing.scale_match_selection_score_margin": result.scale_matcher_config.selection_score_margin,
        "processing.scale_match_maximum_unexplained_weight_fraction": (
            result.scale_matcher_config.maximum_unexplained_weight_fraction
        ),
        "processing.scale_match_maximum_explanation_pitch_class_count": (
            result.scale_matcher_config.maximum_explanation_pitch_class_count
        ),
        "data.parsed_manifest_sha256": file_sha256(result.parsed_manifest_path),
    }
    if result.encoded_manifest_path is not None and result.encoded_manifest_path.exists():
        params["data.encoded_manifest_sha256"] = file_sha256(result.encoded_manifest_path)

    if result.tokenizer_snapshot_path is not None and result.tokenizer_snapshot_path.exists():
        params["data.tokenizer_hash"] = result.tokenizer_snapshot_path.parent.name
        params["data.tokenizer_snapshot_sha256"] = file_sha256(result.tokenizer_snapshot_path)

    encoded_jsonl_path = _encoded_jsonl_path(result)
    if encoded_jsonl_path is not None and encoded_jsonl_path.exists():
        params["data.encoded_samples_fingerprint"] = encoded_samples_jsonl_fingerprint(encoded_jsonl_path)
        params["data.encoded_jsonl_sha256"] = file_sha256(encoded_jsonl_path)

    return params


def _processing_metrics(*, result: ProcessDatasetResult) -> dict[str, float]:
    parsed_count, parsed_success_count = _parsed_manifest_counts(result.parsed_manifest_path)
    metrics = {
        "dataset/overall/count/parsed_files": float(parsed_count),
        "dataset/overall/count/parsed_successes": float(parsed_success_count),
        "dataset/overall/count/parse_errors": float(parsed_count - parsed_success_count),
        "dataset/overall/rate/parse_success": _rate(parsed_success_count, parsed_count),
    }
    if result.encoded_manifest_path is None or not result.encoded_manifest_path.exists():
        return metrics

    encoded_metrics = _encoded_manifest_metrics(result.encoded_manifest_path)
    segment_count = encoded_metrics.segment_count
    metrics.update(
        {
            "dataset/overall/count/segments": float(segment_count),
            "dataset/overall/count/encoded_samples": float(encoded_metrics.encoded_count),
            "dataset/overall/rate/eligible": _rate(encoded_metrics.eligible_count, segment_count),
        }
    )
    metrics.update(encoded_metrics.mean_metrics())
    metrics.update(encoded_metrics.boolean_rate_metrics())
    for reason, count in encoded_metrics.ineligibility_reason_counts.items():
        metrics[f"dataset/ineligibility/count/{reason}"] = float(count)
        metrics[f"dataset/ineligibility/rate/{reason}"] = _rate(count, segment_count)

    return metrics


@dataclass
class _EncodedManifestMetrics:
    segment_count: int
    encoded_count: int
    eligible_count: int
    numeric_sums: dict[EncodedManifestField, float]
    numeric_counts: dict[EncodedManifestField, int]
    boolean_true_counts: dict[EncodedManifestField, int]
    ineligibility_reason_counts: dict[str, int]

    def mean_metrics(self) -> dict[str, float]:
        return {
            _NUMERIC_METRIC_NAMES[field]: _rate(self.numeric_sums[field], self.numeric_counts[field])
            for field in _DIAGNOSTIC_NUMERIC_FIELDS
        }

    def boolean_rate_metrics(self) -> dict[str, float]:
        return {
            _BOOLEAN_METRIC_NAMES[field]: _rate(self.boolean_true_counts[field], self.segment_count)
            for field in _DIAGNOSTIC_BOOLEAN_FIELDS
        }


def _parsed_manifest_counts(path: Path) -> tuple[int, int]:
    parsed_count = 0
    parsed_success_count = 0
    for row in iter_parsed_manifest(path):
        parsed_count += 1
        if row[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value:
            parsed_success_count += 1

    return parsed_count, parsed_success_count


def _encoded_manifest_metrics(path: Path) -> _EncodedManifestMetrics:
    metrics = _empty_encoded_manifest_metrics()
    for row in iter_encoded_manifest(path):
        metrics.segment_count += 1
        if row[EncodedManifestField.ENCODED_LINE] != "":
            metrics.encoded_count += 1
        if row[EncodedManifestField.ELIGIBLE_FOR_TRAINING] == _TRUE_TEXT:
            metrics.eligible_count += 1

        _update_numeric_metrics(metrics, row)
        _update_boolean_metrics(metrics, row)
        _update_ineligibility_reason_counts(metrics, row)

    return metrics


def _empty_encoded_manifest_metrics() -> _EncodedManifestMetrics:
    return _EncodedManifestMetrics(
        segment_count=0,
        encoded_count=0,
        eligible_count=0,
        numeric_sums={field: 0.0 for field in _DIAGNOSTIC_NUMERIC_FIELDS},
        numeric_counts={field: 0 for field in _DIAGNOSTIC_NUMERIC_FIELDS},
        boolean_true_counts={field: 0 for field in _DIAGNOSTIC_BOOLEAN_FIELDS},
        ineligibility_reason_counts={},
    )


def _update_numeric_metrics(metrics: _EncodedManifestMetrics, row: dict[str, str]) -> None:
    for field in _DIAGNOSTIC_NUMERIC_FIELDS:
        if row[field] == "":
            continue

        metrics.numeric_sums[field] += float(row[field])
        metrics.numeric_counts[field] += 1


def _update_boolean_metrics(metrics: _EncodedManifestMetrics, row: dict[str, str]) -> None:
    for field in _DIAGNOSTIC_BOOLEAN_FIELDS:
        if row[field] == _TRUE_TEXT:
            metrics.boolean_true_counts[field] += 1


def _update_ineligibility_reason_counts(metrics: _EncodedManifestMetrics, row: dict[str, str]) -> None:
    reason_text = row[EncodedManifestField.INELIGIBILITY_REASONS]
    if reason_text == "":
        return

    for reason in reason_text.split("|"):
        metrics.ineligibility_reason_counts[reason] = metrics.ineligibility_reason_counts.get(reason, 0) + 1


def _encoded_jsonl_path(result: ProcessDatasetResult) -> Path | None:
    if result.encoded_manifest_path is None:
        return None

    return result.encoded_manifest_path.parent / ENCODED_JSONL_NAME


def _rate(numerator: float, denominator: int) -> float:
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
