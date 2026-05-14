import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from musak_model.data.schema import SegmentMetadata
from musak_model.model.config import CNNConfig, ConditioningConfig, GRUConfig, ModelConfig, TransformerConfig
from musak_model.tokens.schema import ScaleType
from musak_model.training.config import TrainingConfig
from musak_model.training.ingestion.schema import EncodedExercise, IngestionErrorRecord, IngestionSplit
from musak_model.training.tracking import MlflowTrainingTracker, build_training_tracker


class FakeMlflow(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.run_name: str | None = None
        self.ended_status: str | None = None
        self.params: dict[str, str | int | float | bool] = {}
        self.metrics: list[tuple[str, float, int]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.logged_dicts: list[tuple[dict[str, Any], str]] = []

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

    def log_metric(self, key: str, value: float, *, step: int) -> None:
        self.metrics.append((key, value, step))

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        self.logged_dicts.append((dictionary, artifact_file))


def _training_config(tmp_path: Path, *, enable_mlflow: bool = True, tracking_uri: str | None = None) -> TrainingConfig:
    return TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        weight_decay=0.0,
        num_workers=0,
        checkpoint_dir=tmp_path / "checkpoints",
        device="cpu",
        enable_mlflow=enable_mlflow,
        mlflow_tracking_uri=tracking_uri,
        mlflow_run_name="test-run",
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        cnn=CNNConfig(out_channels=16, kernel_sizes=(3,), num_layers=1, dropout=0.0),
        gru=GRUConfig(hidden_size=16, num_layers=1, dropout=0.0, bidirectional=False),
        transformer=TransformerConfig(
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            feedforward_size=32,
            dropout=0.0,
            max_sequence_length=64,
        ),
        conditioning=ConditioningConfig(
            num_difficulty_levels=6,
            num_scale_types=9,
            num_time_signatures=5,
            cfg_dropout_probability=0.0,
        ),
    )


def _split() -> IngestionSplit:
    metadata = SegmentMetadata(
        key_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        bar_count=1,
        window_start_bar=0,
        source_file=Path("piece.mxl"),
        difficulty_level=1,
    )
    sample = EncodedExercise(token_ids=[1, 2], bar_positions=[0, 0], metadata=metadata)
    return IngestionSplit(
        train=[sample],
        validation=[],
        invalid_files=[IngestionErrorRecord(file="bad.mxl", exception_type="ValueError", message="bad")],
    )


def test_build_training_tracker_can_disable_mlflow(tmp_path: Path) -> None:
    tracker = build_training_tracker(training_config=_training_config(tmp_path, enable_mlflow=False))

    assert tracker.__class__.__name__ == "NoOpTrainingTracker"


def test_mlflow_tracker_logs_setup_metrics_artifacts_and_invalid_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint")

    tracker = MlflowTrainingTracker(
        training_config=_training_config(tmp_path, tracking_uri="file:///tmp/mlruns"),
        tracking_root=tmp_path,
    )

    with tracker:
        tracker.log_setup(training_config=_training_config(tmp_path), model_config=_model_config(), split=_split())
        tracker.log_epoch(epoch=3, train_loss=1.25, validation_loss=1.5)
        tracker.log_checkpoints(latest_checkpoint_path=checkpoint, best_checkpoint_path=None)
        tracker.log_invalid_files(invalid_files=_split().invalid_files)

    assert fake_mlflow.tracking_uri == "file:///tmp/mlruns"
    assert fake_mlflow.experiment_name == "musak-stage-one"
    assert fake_mlflow.run_name == "test-run"
    assert fake_mlflow.ended_status == "FINISHED"
    assert fake_mlflow.params["training.epochs"] == 1
    assert fake_mlflow.params["data.train_samples"] == 1
    assert ("train_loss", 1.25, 3) in fake_mlflow.metrics
    assert ("validation_loss", 1.5, 3) in fake_mlflow.metrics
    assert fake_mlflow.artifacts == [(str(checkpoint), "checkpoints")]
    assert fake_mlflow.logged_dicts[0][1] == "invalid_files.json"


def test_mlflow_tracker_uses_environment_uri_before_local_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///env/mlruns")

    with MlflowTrainingTracker(training_config=_training_config(tmp_path), tracking_root=tmp_path):
        pass

    assert fake_mlflow.tracking_uri == "file:///env/mlruns"
