import sys
from pathlib import Path
from types import ModuleType

import pytest

from musak_model.data.schema import SegmentMetadata
from musak_model.processing.dataset import ProcessDatasetResult
from musak_model.processing.io import append_jsonl
from musak_model.processing.manifest import (
    ENCODED_MANIFEST_FIELDS,
    PARSED_MANIFEST_FIELDS,
    EncodedManifestField,
    ParsedManifestField,
    ParsedManifestStatus,
    write_encoded_manifest,
    write_parsed_manifest,
)
from musak_model.processing.tracking import ProcessingMlflowConfig, ProcessingTracker
from musak_model.tokens.schema import ScaleType
from musak_model.training.ingestion.schema import EncodedExercise


class FakeMlflow(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.run_name: str | None = None
        self.ended_status: str | None = None
        self.params: dict[str, str | int | float | bool] = {}
        self.metrics: list[tuple[str, float]] = []
        self.artifacts: list[tuple[str, str | None]] = []

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def start_run(self, *, run_name: str | None = None) -> None:
        self.run_name = run_name

    def end_run(self, *, status: str) -> None:
        self.ended_status = status

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        self.metrics.append((key, value))

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))


def test_processing_tracker_logs_complete_manifest_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    dataset_dir = tmp_path / "data" / "PDMX"
    processed_root = tmp_path / "processed"
    artifact_dir = processed_root / "PDMX"
    encoded_dir = artifact_dir / "encoded" / "abc123"
    parsed_manifest_path = artifact_dir / "parsed.csv"
    encoded_manifest_path = encoded_dir / "encoded.csv"
    tokenizer_snapshot_path = encoded_dir / "tokenizer.json"
    encoded_jsonl_path = encoded_dir / "data-00000.jsonl"
    dataset_dir.mkdir(parents=True)
    tokenizer_snapshot_path.parent.mkdir(parents=True)
    tokenizer_snapshot_path.write_text("{}", encoding="utf-8")
    write_parsed_manifest(
        [
            _parsed_row("ok.mxl", ParsedManifestStatus.SUCCESS),
            _parsed_row("bad.mxl", ParsedManifestStatus.ERROR),
        ],
        parsed_manifest_path,
    )
    write_encoded_manifest(
        [
            _encoded_row("a", eligible=True, empty_score=False, one_hand_only=True, right_silence=0.25),
            _encoded_row("b", eligible=False, empty_score=True, one_hand_only=False, right_silence=1.0),
        ],
        encoded_manifest_path,
    )
    append_jsonl(_encoded_sample("ok.mxl", [1, 2]), encoded_jsonl_path)
    result = ProcessDatasetResult(
        parsed_manifest_path=parsed_manifest_path,
        encoded_manifest_path=encoded_manifest_path,
        tokenizer_snapshot_path=tokenizer_snapshot_path,
        parsed_count=1,
        encoded_count=1,
        error_count=1,
    )

    with ProcessingTracker(
        config=ProcessingMlflowConfig(
            experiment_name="musak-process-test",
            run_name="process-run",
            tracking_uri="file:///tmp/mlruns",
        ),
        tracking_root=tmp_path,
    ) as tracker:
        tracker.log_processing_result(
            result=result,
            data_dir=dataset_dir,
            processed_root=processed_root,
            stage="all",
            overwrite=False,
        )

    metrics = dict(fake_mlflow.metrics)
    assert fake_mlflow.tracking_uri == "file:///tmp/mlruns"
    assert fake_mlflow.experiment_name == "musak-process-test"
    assert fake_mlflow.run_name == "process-run"
    assert fake_mlflow.ended_status == "FINISHED"
    assert fake_mlflow.params["data.dataset_name"] == "PDMX"
    assert fake_mlflow.params["processing.overwrite"] is False
    assert fake_mlflow.params["data.tokenizer_hash"] == "abc123"
    assert "data.encoded_samples_fingerprint" in fake_mlflow.params
    assert metrics["dataset/overall/count/parsed_files"] == 2.0
    assert metrics["dataset/overall/rate/parse_success"] == 0.5
    assert metrics["dataset/overall/count/segments"] == 2.0
    assert metrics["dataset/overall/rate/eligible"] == 0.5
    assert metrics["dataset/ineligibility/count/overlapping_events"] == 1.0
    assert metrics["dataset/ineligibility/rate/overlapping_events"] == 0.5
    assert metrics["dataset/ineligibility/count/quantization_error"] == 1.0
    assert metrics["dataset/diagnostics/rate/empty_score"] == 0.5
    assert metrics["dataset/diagnostics/mean/right_silence_fraction"] == 0.625
    assert metrics["dataset/diagnostics/mean/silent_bar_count"] == 1.0
    assert metrics["dataset/diagnostics/mean/silent_bar_fraction"] == 0.5
    assert metrics["dataset/diagnostics/mean/silent_edge_bar_count"] == 0.0
    assert (str(encoded_manifest_path), "dataset") in fake_mlflow.artifacts


def _parsed_row(source_path: str, status: ParsedManifestStatus) -> dict[object, object]:
    row = {field: "" for field in PARSED_MANIFEST_FIELDS}
    row.update(
        {
            ParsedManifestField.SOURCE_ID: source_path,
            ParsedManifestField.SOURCE_PATH: source_path,
            ParsedManifestField.STATUS: status.value,
            ParsedManifestField.ERROR_TYPE: "ValueError" if status == ParsedManifestStatus.ERROR else "",
            ParsedManifestField.ERROR_MESSAGE: "bad" if status == ParsedManifestStatus.ERROR else "",
        }
    )
    return row


def _encoded_row(
    segment_id: str,
    *,
    eligible: bool,
    empty_score: bool,
    one_hand_only: bool,
    right_silence: float,
) -> dict[object, object]:
    row = {field: "" for field in ENCODED_MANIFEST_FIELDS}
    row.update(
        {
            EncodedManifestField.SEGMENT_ID: segment_id,
            EncodedManifestField.ENCODED_LINE: "0" if eligible else "",
            EncodedManifestField.ELIGIBLE_FOR_TRAINING: str(eligible),
            EncodedManifestField.INELIGIBILITY_REASONS: "" if eligible else "overlapping_events|quantization_error",
            EncodedManifestField.EMPTY_SCORE: str(empty_score),
            EncodedManifestField.ONE_HAND_ONLY: str(one_hand_only),
            EncodedManifestField.RIGHT_SILENCE_FRACTION: right_silence,
            EncodedManifestField.LEFT_SILENCE_FRACTION: 1.0,
            EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION: 0.5,
            EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION: 0.0,
            EncodedManifestField.HAND_ACTIVITY_BALANCE: 0.25,
            EncodedManifestField.SILENT_BAR_COUNT: 1.0,
            EncodedManifestField.SILENT_BAR_FRACTION: 0.5,
            EncodedManifestField.SILENT_EDGE_BAR_COUNT: 0.0,
            EncodedManifestField.NOTE_TOKEN_FRACTION: 0.25,
            EncodedManifestField.REST_TOKEN_FRACTION: 0.5,
            EncodedManifestField.HOLD_TOKEN_FRACTION: 0.0,
        }
    )
    return row


def _encoded_sample(source_file: str, token_ids: list[int]) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=[0 for _ in token_ids],
        metadata=SegmentMetadata(
            key_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path(source_file),
            difficulty_level=None,
        ),
    )
