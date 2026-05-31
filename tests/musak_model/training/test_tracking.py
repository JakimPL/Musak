import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from musak_model.conditioning.config import ConditioningConfig, DifficultyConfig
from musak_model.conditioning.time_signature import TimeSignatureVocabularyConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.model.config import (
    CNNConfig,
    GRUConfig,
    ModelConfig,
    ModelOutputConfig,
    ModelOutputMode,
    TransformerConfig,
)
from musak_model.tokens.schema import ScaleType
from musak_model.training.config import (
    CheckpointConfig,
    EventObjectiveConfig,
    GenerationEvaluationConfig,
    MlflowConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainingConditioningConfig,
    TrainingConfig,
)
from musak_model.training.ingestion.schema import EncodedExercise, IngestionErrorRecord, IngestionSplit
from musak_model.training.metrics import EpochMetrics
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
        optimization=OptimizationConfig(epochs=1, batch_size=2, learning_rate=0.001, weight_decay=0.0),
        event_objective=_event_objective_config(),
        runtime=RuntimeConfig(num_workers=1, device="cpu"),
        conditioning=TrainingConditioningConfig(
            use_time_signature=False,
            use_scale_type=False,
            use_difficulty=False,
            use_structural_conditioning=False,
            use_validity_penalty=False,
            validity_penalty_weight=0.05,
        ),
        checkpoints=CheckpointConfig(checkpoint_directory=tmp_path / "checkpoints"),
        mlflow=MlflowConfig(
            enable_mlflow=enable_mlflow,
            mlflow_tracking_uri=tracking_uri,
            mlflow_run_name="test-run",
        ),
        generation_evaluation=GenerationEvaluationConfig(
            enabled=False,
            every_epochs=5,
            soft_sample_count=4,
            hard_sample_count=4,
            max_new_tokens=256,
            temperature=1.0,
            top_k=32,
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            minimum_duration_denominator=16,
            allow_dotted_durations=True,
            max_notes_per_hand=5,
            maximum_onset_span_semitones=12,
            maximum_pitch_gap_semitones=12,
            maximum_static_hand_span_degrees=5,
        ),
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocabulary_size=32,
        duration_vocabulary_size=1,
        output=ModelOutputConfig(mode=ModelOutputMode.FLAT),
        cnn=CNNConfig(enabled=True, out_channels=16, kernel_sizes=(3,), num_layers=1, dropout=0.0),
        gru=GRUConfig(enabled=True, hidden_size=16, num_layers=1, dropout=0.0, bidirectional=False),
        transformer=TransformerConfig(
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            feedforward_size=32,
            dropout=0.0,
            max_sequence_length=64,
        ),
        conditioning=ConditioningConfig(
            difficulty=DifficultyConfig(max_level=5),
            time_signature=TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2),
            cfg_dropout_probability=0.0,
        ),
    )


def _event_objective_config() -> EventObjectiveConfig:
    return EventObjectiveConfig(
        mode=ModelOutputMode.FLAT,
        kind_weight=1.0,
        duration_weight=1.0,
        degree_weight=1.0,
        accidental_weight=1.0,
        octave_offset_weight=1.0,
        hand_weight=1.0,
    )


def _split() -> IngestionSplit:
    metadata = SegmentMetadata(
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
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
        tracker.log_epoch(
            metrics=EpochMetrics(
                epoch=3,
                train_loss=1.25,
                train_perplexity=3.49,
                train_token_accuracy=0.5,
                train_token_kind_accuracy=0.75,
                train_cnn_gradient_norm=0.1,
                train_gru_gradient_norm=0.2,
                train_transformer_gradient_norm=0.3,
                validation_loss=1.5,
                validation_perplexity=4.48,
                validation_token_accuracy=0.4,
                validation_token_kind_accuracy=0.6,
            )
        )
        tracker.log_generation_evaluation(metrics={"generation/soft/rate/end": 0.25}, epoch=3)
        tracker.log_split_figure_metrics(metrics={"model/split/figure/count/comparable_groups": 1.0})
        tracker.log_checkpoints(latest_checkpoint_path=checkpoint, best_checkpoint_path=None)
        tracker.log_invalid_files(invalid_files=_split().invalid_files)

    assert fake_mlflow.tracking_uri == "file:///tmp/mlruns"
    assert fake_mlflow.experiment_name == "musak-pretrain"
    assert fake_mlflow.run_name == "test-run"
    assert fake_mlflow.ended_status == "FINISHED"
    assert fake_mlflow.params["training.optimization.epochs"] == 1
    assert fake_mlflow.params["data.train_samples"] == 1
    assert "data.encoded_samples_fingerprint" in fake_mlflow.params
    assert ("model/train/mean/loss", 1.25, 3) in fake_mlflow.metrics
    assert ("model/train/mean/perplexity", 3.49, 3) in fake_mlflow.metrics
    assert ("model/train/rate/token_accuracy", 0.5, 3) in fake_mlflow.metrics
    assert ("model/train/rate/token_kind_accuracy", 0.75, 3) in fake_mlflow.metrics
    assert ("model/train/mean/cnn_gradient_norm", 0.1, 3) in fake_mlflow.metrics
    assert ("model/train/mean/gru_gradient_norm", 0.2, 3) in fake_mlflow.metrics
    assert ("model/train/mean/transformer_gradient_norm", 0.3, 3) in fake_mlflow.metrics
    assert ("model/validation/mean/loss", 1.5, 3) in fake_mlflow.metrics
    assert ("model/validation/mean/perplexity", 4.48, 3) in fake_mlflow.metrics
    assert ("model/validation/rate/token_accuracy", 0.4, 3) in fake_mlflow.metrics
    assert ("model/validation/rate/token_kind_accuracy", 0.6, 3) in fake_mlflow.metrics
    assert ("generation/soft/rate/end", 0.25, 3) in fake_mlflow.metrics
    assert ("model/split/figure/count/comparable_groups", 1.0, 0) in fake_mlflow.metrics
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


def test_mlflow_tracker_uses_sqlite_local_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with MlflowTrainingTracker(training_config=_training_config(tmp_path), tracking_root=tmp_path):
        pass

    assert fake_mlflow.tracking_uri == f"sqlite:///{tmp_path / 'artifacts/mlflow/mlflow.db'}"
    assert (tmp_path / "artifacts/mlflow").exists()
