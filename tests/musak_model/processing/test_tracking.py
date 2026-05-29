import sys
from pathlib import Path
from types import ModuleType

import pytest

from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.n_grams.profile.extraction import FigureExtractionResult
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
    encoded_directory = artifact_dir / "encoded" / "abc123"
    parsed_manifest_path = artifact_dir / "parsed.csv"
    encoded_manifest_path = encoded_directory / "encoded.csv"
    tokenizer_snapshot_path = encoded_directory / "tokenizer.json"
    encoded_jsonl_path = encoded_directory / "data-00000.jsonl"
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
        scale_matcher_config=ScaleMatcherConfig(
            support_score_margin=0.08,
            selection_score_margin=0.03,
            maximum_unexplained_weight_fraction=0.10,
            maximum_explanation_pitch_class_count=9,
        ),
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
            stage="process",
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
    assert metrics["dataset/diagnostics/mean/accidental_note_fraction"] == 0.125
    assert metrics["dataset/diagnostics/mean/in_scale_note_fraction"] == 0.875
    assert metrics["dataset/diagnostics/mean/note_density_per_beat"] == 0.5
    assert metrics["dataset/diagnostics/rate/has_dotted_notes"] == 0.5
    assert metrics["dataset/diagnostics/mean/max_notes_per_onset"] == 1.5
    assert metrics["dataset/diagnostics/mean/synchronized_onset_fraction"] == 0.25
    assert metrics["dataset/scale_match/mean/in_scale_weight_fraction"] == 0.8
    assert metrics["dataset/scale_match/mean/unexplained_out_of_scale_weight_fraction"] == 0.05
    assert metrics["dataset/scale_match/mean/explanation_pitch_class_count"] == 8.0
    assert metrics["dataset/scale_match/rate/declared_match_used"] == 0.5
    assert (str(encoded_manifest_path), "dataset") in fake_mlflow.artifacts


def test_processing_tracker_uses_sqlite_local_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with ProcessingTracker(config=ProcessingMlflowConfig(), tracking_root=tmp_path):
        pass

    assert fake_mlflow.tracking_uri == f"sqlite:///{tmp_path / 'artifacts/mlflow/mlflow.db'}"
    assert (tmp_path / "artifacts/mlflow").exists()


def test_processing_tracker_logs_figure_artifacts_in_same_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    encoded_directory = tmp_path / "processed" / "PDMX" / "encoded" / "abc123"
    figure_dir = encoded_directory / "figure"
    all_dir = figure_dir / "all"
    config_path = figure_dir / "config.yml"
    counts_path = all_dir / "counts.parquet"
    base_durations_path = all_dir / "base_durations.parquet"
    profile_path = all_dir / "profile.json"
    by_sample_path = figure_dir / "by_sample.jsonl"
    extra_output_path = tmp_path / "analysis" / "figures.csv"
    for path in (config_path, counts_path, base_durations_path, profile_path, by_sample_path, extra_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    figure_result = FigureExtractionResult(
        artifact_paths=FigureArtifactPaths(
            root_directory=figure_dir,
            config_path=config_path,
            all_directory=all_dir,
            profile_path=profile_path,
            counts_path=counts_path,
            base_durations_path=base_durations_path,
            by_sample_path=by_sample_path,
        ),
        encoded_sample_count=12,
        profile_group_count=4,
        sample_profile_count=12,
        extra_output_path=extra_output_path,
    )

    with ProcessingTracker(config=ProcessingMlflowConfig(run_name="process-run"), tracking_root=tmp_path) as tracker:
        tracker.log_figure_extraction_result(figure_result)

    metrics = dict(fake_mlflow.metrics)
    assert fake_mlflow.run_name == "process-run"
    assert fake_mlflow.ended_status == "FINISHED"
    assert metrics["dataset/figure/count/encoded_samples"] == 12.0
    assert metrics["dataset/figure/count/profile_groups"] == 4.0
    assert metrics["dataset/figure/count/sample_profiles"] == 12.0
    assert (str(config_path), "dataset/figure") in fake_mlflow.artifacts
    assert (str(by_sample_path), "dataset/figure") in fake_mlflow.artifacts
    assert (str(counts_path), "dataset/figure/all") in fake_mlflow.artifacts
    assert (str(profile_path), "dataset/figure/all") in fake_mlflow.artifacts
    assert (str(extra_output_path), "dataset/figure/extra") in fake_mlflow.artifacts


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
            EncodedManifestField.ACCIDENTAL_NOTE_FRACTION: 0.25 if eligible else 0.0,
            EncodedManifestField.IN_SCALE_NOTE_FRACTION: 0.75 if eligible else 1.0,
            EncodedManifestField.NOTE_DENSITY_PER_BEAT: 0.25 if eligible else 0.75,
            EncodedManifestField.ONSET_DENSITY_PER_BEAT: 0.5,
            EncodedManifestField.RIGHT_ONSET_DENSITY_PER_BEAT: 0.25,
            EncodedManifestField.LEFT_ONSET_DENSITY_PER_BEAT: 0.25,
            EncodedManifestField.SHORTEST_NOTE_DURATION_BEATS: 1.0,
            EncodedManifestField.HAS_DOTTED_NOTES: str(eligible),
            EncodedManifestField.MAX_NOTES_PER_ONSET: 1 if eligible else 2,
            EncodedManifestField.MAX_NOTES_PER_HAND: 1 if eligible else 2,
            EncodedManifestField.MAX_ONSET_SPAN_SEMITONES: 0 if eligible else 7,
            EncodedManifestField.MAX_MELODIC_GAP_SEMITONES: 2 if eligible else 4,
            EncodedManifestField.STATIC_HAND_SPAN_DEGREES: 3 if eligible else 5,
            EncodedManifestField.SYNCHRONIZED_ONSET_FRACTION: 0.5 if eligible else 0.0,
            EncodedManifestField.INDEPENDENT_ONSET_FRACTION: 0.5 if eligible else 1.0,
            EncodedManifestField.SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION: 0.8,
            EncodedManifestField.SCALE_MATCH_OUT_OF_SCALE_WEIGHT_FRACTION: 0.2,
            EncodedManifestField.SCALE_MATCH_EXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION: 0.15,
            EncodedManifestField.SCALE_MATCH_UNEXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION: 0.05,
            EncodedManifestField.SCALE_MATCH_BEST_MARGIN: 0.1,
            EncodedManifestField.SCALE_MATCH_OBSERVED_PITCH_CLASS_COUNT: 5,
            EncodedManifestField.SCALE_MATCH_EXPLANATION_PITCH_CLASS_COUNT: 8,
            EncodedManifestField.SCALE_MATCH_SUPPORT_CANDIDATE_COUNT: 3,
            EncodedManifestField.SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT: 1,
            EncodedManifestField.SCALE_MATCH_DECLARED_MATCH_USED: str(eligible),
            EncodedManifestField.SCALE_MATCH_LOW_CONFIDENCE: str(not eligible),
            EncodedManifestField.SCALE_MATCH_AMBIGUOUS: str(not eligible),
            EncodedManifestField.SCALE_MATCH_NO_PITCHES: str(False),
        }
    )
    return row


def _encoded_sample(source_file: str, token_ids: list[int]) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=[0 for _ in token_ids],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path(source_file),
            difficulty_level=None,
        ),
    )
