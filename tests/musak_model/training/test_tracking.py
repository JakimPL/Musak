import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.conditioning.config import (
    ConditioningConfig,
    DifficultyConfig,
    HarmonicConditioningConfig,
    HarmonicFusionMode,
)
from musak_model.conditioning.time_signature import TimeSignatureVocabularyConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.mlflow import read_mlflow_run_id
from musak_model.model.config import (
    CNNConfig,
    GRUConfig,
    ModelConfig,
    ModelInputConfig,
    ModelOutputConfig,
    ModelOutputMode,
    TokenInputEmbeddingMode,
    TransformerConfig,
)
from musak_model.tokens.schema import ScaleType
from musak_model.training.config import (
    CheckpointConfig,
    EarlyStoppingConfig,
    EventObjectiveConfig,
    GenerationEvaluationConfig,
    HarmonicRelationObjectiveConfig,
    MlflowConfig,
    MusicalAuxiliaryObjectiveConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainingConditioningConfig,
    TrainingConfig,
)
from musak_model.training.ingestion.schema import EncodedExercise, IngestionErrorRecord, IngestionSplit
from musak_model.training.metrics import EpochMetrics
from musak_model.training.tracking import MlflowTrainingTracker, build_training_tracker


class FakeRunInfo:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class FakeActiveRun:
    def __init__(self, run_id: str) -> None:
        self.info = FakeRunInfo(run_id)


class FakeMlflow(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.run_name: str | None = None
        self.run_id: str | None = None
        self.ended_status: str | None = None
        self.params: dict[str, str | int | float | bool] = {}
        self.metrics: list[tuple[str, float, int]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.artifact_directories: list[tuple[str, str | None]] = []
        self.logged_dicts: list[tuple[dict[str, Any], str]] = []
        self.tags: dict[str, str] = {}

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def start_run(self, *, run_id: str | None = None, run_name: str | None = None) -> FakeActiveRun:
        self.run_id = run_id or "generated-run-id"
        self.run_name = run_name
        return FakeActiveRun(self.run_id)

    def end_run(self, *, status: str) -> None:
        self.ended_status = status

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value
        if key == "mlflow.runName":
            self.run_name = value

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float, *, step: int) -> None:
        self.metrics.append((key, value, step))

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))

    def log_artifacts(self, local_dir: str, *, artifact_path: str | None = None) -> None:
        self.artifact_directories.append((local_dir, artifact_path))

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        self.logged_dicts.append((dictionary, artifact_file))


def _training_config(
    tmp_path: Path,
    *,
    enable_mlflow: bool = True,
    tracking_uri: str | None = None,
    run_id: str | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        optimization=OptimizationConfig(epochs=1, batch_size=2, learning_rate=0.001, weight_decay=0.0),
        event_objective=_event_objective_config(),
        musical_auxiliary_targets=_musical_auxiliary_target_config(),
        musical_auxiliary_objective=_musical_auxiliary_objective_config(),
        harmonic_relation_objective=_harmonic_relation_objective_config(),
        early_stopping=EarlyStoppingConfig(enabled=False, patience_epochs=10, min_delta=0.0),
        runtime=RuntimeConfig(num_workers=1, device="cpu"),
        conditioning=TrainingConditioningConfig(
            use_time_signature=False,
            use_scale_type=False,
            use_difficulty=False,
            use_structural_conditioning=False,
            use_harmony_conditioning=False,
            use_validity_penalty=False,
            validity_penalty_weight=0.05,
        ),
        checkpoints=CheckpointConfig(checkpoint_directory=tmp_path / "checkpoints"),
        mlflow=MlflowConfig(
            enable_mlflow=enable_mlflow,
            mlflow_tracking_uri=tracking_uri,
            mlflow_run_name="test-run",
            mlflow_run_id=run_id,
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


def _training_config_with_generated_run_name(tmp_path: Path) -> TrainingConfig:
    return _training_config(tmp_path).model_copy(
        update={
            "mlflow": MlflowConfig(
                enable_mlflow=True,
                mlflow_tracking_uri="file:///tmp/mlruns",
                mlflow_run_name=None,
            )
        }
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocabulary_size=32,
        duration_vocabulary_size=1,
        input=ModelInputConfig(embedding_mode=TokenInputEmbeddingMode.FLAT),
        output=ModelOutputConfig(mode=ModelOutputMode.FLAT),
        musical_auxiliary_targets=_musical_auxiliary_target_config(),
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
            harmony=_harmony_config(enabled=False),
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


def _harmony_config(*, enabled: bool) -> HarmonicConditioningConfig:
    return HarmonicConditioningConfig(
        enabled=enabled,
        fusion=HarmonicFusionMode.GATED_RESIDUAL,
        plan_encoder_layers=1,
        plan_encoder_heads=2,
        plan_encoder_dropout=0.0,
        gate_init_bias=-1.5,
        harmony_adherence_alpha=1.0,
        plan_field_dropout=0.0,
    )


def _musical_auxiliary_objective_config() -> MusicalAuxiliaryObjectiveConfig:
    return MusicalAuxiliaryObjectiveConfig(
        enabled=True,
        weight=0.1,
        bar_weight=1.0,
        note_density_weight=1.0,
        rhythmic_diversity_weight=1.0,
        voice_independence_weight=1.0,
        uses_accidentals_weight=1.0,
        dotted_duration_weight=1.0,
        hand_span_weight=1.0,
    )


def _harmonic_relation_objective_config() -> HarmonicRelationObjectiveConfig:
    return HarmonicRelationObjectiveConfig(
        enabled=True,
        weight=0.03,
        downbeat_weight=1.5,
        strong_beat_weight=1.2,
        weak_beat_weight=0.7,
        left_hand_weight=1.2,
        right_hand_weight=1.0,
        opening_weight=1.0,
        continuation_weight=1.0,
        cadence_preparation_weight=1.2,
        cadence_weight=1.5,
        use_plan_confidence_weight=True,
        minimum_plan_confidence_weight=0.5,
    )


def _musical_auxiliary_target_config() -> MusicalAuxiliaryTargetConfig:
    return MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
        rhythmic_diversity_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        voice_independence_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        hand_span_bucket_boundaries=(3, 5, 8, 12, 16),
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

    assert isinstance(tracker, MlflowTrainingTracker)
    assert not tracker.enabled


def test_mlflow_tracker_logs_setup_metrics_artifacts_and_invalid_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint")
    generation_artifact_directory = tmp_path / "generation-artifacts"
    generation_artifact_directory.mkdir()
    (generation_artifact_directory / "samples.jsonl").write_text("", encoding="utf-8")

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
                train_musical_auxiliary_loss=2.5,
                train_note_density_accuracy=0.8,
                train_bar_note_density_accuracy=0.7,
                train_harmonic_relation_target_distribution=(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
                train_harmonic_relation_prediction_distribution=(0.25, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0),
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
        tracker.log_generation_artifacts(artifact_directory=generation_artifact_directory, epoch=3)
        tracker.log_split_figure_metrics(metrics={"model/split/figure/count/comparable_groups": 1.0})
        tracker.log_checkpoints(latest_checkpoint_path=checkpoint, best_checkpoint_path=None)
        tracker.log_invalid_files(invalid_files=_split().invalid_files)

    assert fake_mlflow.tracking_uri == "file:///tmp/mlruns"
    assert fake_mlflow.experiment_name == "musak-pretrain"
    assert fake_mlflow.run_name == "test-run"
    assert fake_mlflow.ended_status == "FINISHED"
    assert fake_mlflow.params["training.optimization.epochs"] == 1
    assert "training.checkpoints.resume_checkpoint" not in fake_mlflow.params
    assert "training.mlflow.mlflow_run_id" not in fake_mlflow.params
    assert fake_mlflow.params["data.train_samples"] == 1
    assert "data.encoded_samples_fingerprint" in fake_mlflow.params
    assert ("model/train/mean/loss", 1.25, 3) in fake_mlflow.metrics
    assert ("model/train/mean/perplexity", 3.49, 3) in fake_mlflow.metrics
    assert ("model/train/rate/token_accuracy", 0.5, 3) in fake_mlflow.metrics
    assert ("model/train/rate/token_kind_accuracy", 0.75, 3) in fake_mlflow.metrics
    assert ("model/train/mean/musical_auxiliary_loss", 2.5, 3) in fake_mlflow.metrics
    assert ("model/train/rate/note_density_accuracy", 0.8, 3) in fake_mlflow.metrics
    assert ("model/train/rate/bar_note_density_accuracy", 0.7, 3) in fake_mlflow.metrics
    assert ("model/train/distribution/harmonic_relation/target/chord_root", 0.5, 3) in fake_mlflow.metrics
    assert ("model/train/distribution/harmonic_relation/prediction/chord_third", 0.75, 3) in fake_mlflow.metrics
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
    assert fake_mlflow.artifact_directories == [(str(generation_artifact_directory), "generation/epoch_0003")]
    assert fake_mlflow.logged_dicts[0][1] == "invalid_files.json"
    assert read_mlflow_run_id(tmp_path / "checkpoints") == "generated-run-id"


def test_mlflow_tracker_attaches_to_existing_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    training_config = _training_config(tmp_path, tracking_uri="file:///tmp/mlruns", run_id="abc123").model_copy(
        update={
            "checkpoints": CheckpointConfig(
                checkpoint_directory=tmp_path / "checkpoints",
                resume_checkpoint=tmp_path / "checkpoints" / "latest.pt",
            )
        }
    )

    with MlflowTrainingTracker(training_config=training_config, tracking_root=tmp_path) as tracker:
        tracker.log_setup(training_config=training_config, model_config=_model_config(), split=_split())

    assert fake_mlflow.run_id == "abc123"
    assert fake_mlflow.run_name is None
    assert "training.checkpoints.resume_checkpoint" not in fake_mlflow.params
    assert "mlflow.runName" not in fake_mlflow.tags
    assert read_mlflow_run_id(tmp_path / "checkpoints") == "abc123"


def test_mlflow_tracker_generates_informative_default_run_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    training_config = _training_config_with_generated_run_name(tmp_path)

    with MlflowTrainingTracker(training_config=training_config, tracking_root=tmp_path) as tracker:
        assert fake_mlflow.run_name == "pretrain-flat-e1-bs2-lr0p001-cpu-noes-aux0p1-vp0p05-gen2b-4s4h"
        tracker.log_setup(training_config=training_config, model_config=_model_config(), split=_split())

    assert fake_mlflow.run_name.startswith("pretrain-flat-e1-bs2-lr0p001-cpu-noes-aux0p1-vp0p05-gen2b-4s4h")
    assert "-tr1-va0-bad1-fp" in fake_mlflow.run_name
    assert fake_mlflow.tags["mlflow.runName"] == fake_mlflow.run_name


def test_mlflow_tracker_keeps_explicit_run_name_after_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    training_config = _training_config(tmp_path, tracking_uri="file:///tmp/mlruns")

    with MlflowTrainingTracker(training_config=training_config, tracking_root=tmp_path) as tracker:
        tracker.log_setup(training_config=training_config, model_config=_model_config(), split=_split())

    assert fake_mlflow.run_name == "test-run"
    assert "mlflow.runName" not in fake_mlflow.tags


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
