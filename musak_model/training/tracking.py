import logging
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Final, Protocol, Self

from pydantic import BaseModel

from musak_model.conditioning.harmony.relations import HarmonicRelationId
from musak_model.mlflow import MlflowRun, MlflowRunConfig, flatten_params, write_mlflow_run_id
from musak_model.model.config import ModelConfig
from musak_model.processing.fingerprint import encoded_samples_fingerprint
from musak_model.training.config import TrainingConfig
from musak_model.training.ingestion.schema import IngestionErrorRecord, IngestionSplit
from musak_model.training.metrics import EpochMetrics

_LOGGER = logging.getLogger(__name__)
_MLFLOW_RUN_NAME_TAG: Final[str] = "mlflow.runName"
_MUSAK_EXPERIMENT_PREFIX: Final[str] = "musak-"
_FINGERPRINT_NAME_LENGTH: Final[int] = 8
_GENERATION_ARTIFACT_ROOT: Final[str] = "generation"


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

    def log_generation_artifacts(self, *, artifact_directory: Path, epoch: int) -> None: ...

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

    def log_generation_artifacts(self, *, artifact_directory: Path, epoch: int) -> None:
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
        self._uses_generated_run_name = (
            training_config.mlflow.mlflow_run_name is None and training_config.mlflow.mlflow_run_id is None
        )
        self._run_name = training_config.mlflow.mlflow_run_name or _mlflow_run_name(training_config)
        self._run = MlflowRun(
            MlflowRunConfig(
                enabled=training_config.mlflow.enable_mlflow,
                experiment_name=training_config.mlflow.mlflow_experiment_name,
                run_name=self._run_name,
                run_id=training_config.mlflow.mlflow_run_id,
                tracking_uri=training_config.mlflow.mlflow_tracking_uri,
            ),
            tracking_root=tracking_root,
        )

    @property
    def enabled(self) -> bool:
        return self._run.enabled

    def __enter__(self) -> Self:
        self._run.__enter__()
        if self._run.run_id is not None:
            write_mlflow_run_id(
                checkpoint_directory=self._training_config.checkpoints.checkpoint_directory,
                run_id=self._run.run_id,
            )
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
        if not self._run.enabled:
            return

        _LOGGER.info(
            "Logging MLflow training setup: train_samples=%s validation_samples=%s invalid_files=%s",
            len(split.train),
            len(split.validation),
            len(split.invalid_files),
        )
        started_at = perf_counter()
        _LOGGER.info("Computing encoded sample fingerprint for MLflow setup")
        fingerprint = encoded_samples_fingerprint([*split.train, *split.validation])
        if self._uses_generated_run_name:
            self._run_name = _mlflow_run_name_with_split(
                training_config=training_config,
                split=split,
                fingerprint=fingerprint,
            )
            self._run.set_tag(_MLFLOW_RUN_NAME_TAG, self._run_name)

        params = flatten_params(
            {
                "training": _training_config_param_dump(training_config),
                "model": _serializable_dump(model_config),
                "data": {
                    "train_samples": len(split.train),
                    "validation_samples": len(split.validation),
                    "invalid_files": len(split.invalid_files),
                    "encoded_samples_fingerprint": fingerprint,
                },
            }
        )
        self._run.log_params(params)
        _LOGGER.info("Logged MLflow training setup in %.1fs", perf_counter() - started_at)

    def log_epoch(
        self,
        *,
        metrics: EpochMetrics,
    ) -> None:
        self._run.log_metrics(_epoch_metric_values(metrics), step=metrics.epoch)

    def log_generation_evaluation(self, *, metrics: dict[str, float], epoch: int) -> None:
        if not self._run.enabled:
            return

        _LOGGER.info("Logging %s generation evaluation metric(s) to MLflow", len(metrics))
        self._run.log_metrics(metrics, step=epoch)

    def log_generation_artifacts(self, *, artifact_directory: Path, epoch: int) -> None:
        if not self._run.enabled:
            return

        artifact_path = f"{_GENERATION_ARTIFACT_ROOT}/epoch_{epoch:04d}"
        _LOGGER.info("Logging generation evaluation artifacts to MLflow: artifact_path=%s", artifact_path)
        self._run.log_artifacts(artifact_directory, artifact_path=artifact_path)

    def log_split_figure_metrics(self, *, metrics: dict[str, float]) -> None:
        if not self._run.enabled:
            return

        _LOGGER.info("Logging %s split figure metric(s) to MLflow", len(metrics))
        self._run.log_metrics(metrics, step=0)

    def log_checkpoints(
        self,
        *,
        latest_checkpoint_path: Path | None,
        best_checkpoint_path: Path | None,
    ) -> None:
        if not self._run.enabled:
            return

        if latest_checkpoint_path is not None and latest_checkpoint_path.exists():
            _LOGGER.info("Logging latest checkpoint artifact to MLflow: %s", latest_checkpoint_path)
            self._run.log_artifact(latest_checkpoint_path, artifact_path="checkpoints")

        if best_checkpoint_path is not None and best_checkpoint_path.exists():
            _LOGGER.info("Logging best checkpoint artifact to MLflow: %s", best_checkpoint_path)
            self._run.log_artifact(best_checkpoint_path, artifact_path="checkpoints")

    def log_invalid_files(self, *, invalid_files: list[IngestionErrorRecord]) -> None:
        if not self._run.enabled:
            return

        if invalid_files:
            _LOGGER.info("Logging invalid file report to MLflow: invalid_files=%s", len(invalid_files))
            self._run.log_dict(
                {"invalid_files": [_serializable_dump(record) for record in invalid_files]},
                "invalid_files.json",
            )


def build_training_tracker(
    *,
    training_config: TrainingConfig,
    tracking_root: Path | None = None,
) -> TrainingTracker:
    return MlflowTrainingTracker(training_config=training_config, tracking_root=tracking_root)


def _serializable_dump(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="json")


def _training_config_param_dump(training_config: TrainingConfig) -> dict[str, object]:
    return training_config.model_dump(
        mode="json",
        exclude={
            "checkpoints": {"resume_checkpoint"},
            "mlflow": {"mlflow_run_id"},
        },
    )


def _mlflow_run_name(training_config: TrainingConfig) -> str:
    generation = training_config.generation_evaluation
    return "-".join(
        (
            _stage_name(training_config),
            training_config.event_objective.mode.value,
            f"e{training_config.optimization.epochs}",
            f"bs{training_config.optimization.batch_size}",
            f"lr{_format_run_name_number(training_config.optimization.learning_rate)}",
            training_config.runtime.device,
            _early_stopping_run_name_part(training_config),
            f"aux{_format_run_name_number(training_config.musical_auxiliary_objective.weight)}",
            f"vp{_format_run_name_number(training_config.conditioning.validity_penalty_weight)}",
            f"gen{generation.bar_count}b-{generation.soft_sample_count}s{generation.hard_sample_count}h",
        )
    )


def _mlflow_run_name_with_split(
    *,
    training_config: TrainingConfig,
    split: IngestionSplit,
    fingerprint: str,
) -> str:
    return "-".join(
        (
            _mlflow_run_name(training_config),
            f"tr{len(split.train)}",
            f"va{len(split.validation)}",
            f"bad{len(split.invalid_files)}",
            f"fp{fingerprint[:_FINGERPRINT_NAME_LENGTH]}",
        )
    )


def _stage_name(training_config: TrainingConfig) -> str:
    experiment_name = training_config.mlflow.mlflow_experiment_name
    if experiment_name.startswith(_MUSAK_EXPERIMENT_PREFIX):
        return experiment_name.removeprefix(_MUSAK_EXPERIMENT_PREFIX)

    return experiment_name


def _early_stopping_run_name_part(training_config: TrainingConfig) -> str:
    early_stopping = training_config.early_stopping
    if not early_stopping.enabled:
        return "noes"

    return f"es{early_stopping.patience_epochs}"


def _format_run_name_number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _epoch_metric_values(metrics: EpochMetrics) -> dict[str, float]:
    values = {
        metric_name: value
        for field_name, metric_name in _EPOCH_METRIC_NAME_MAP.items()
        if isinstance((value := getattr(metrics, field_name)), float)
    }
    values.update(_harmonic_relation_distribution_metric_values(metrics))
    return values


def _harmonic_relation_distribution_metric_values(metrics: EpochMetrics) -> dict[str, float]:
    values: dict[str, float] = {}
    for split_name in ("train", "validation"):
        target_distribution = getattr(metrics, f"{split_name}_harmonic_relation_target_distribution")
        prediction_distribution = getattr(metrics, f"{split_name}_harmonic_relation_prediction_distribution")
        values.update(
            _named_distribution_metrics(
                target_distribution,
                metric_prefix=f"model/{split_name}/distribution/harmonic_relation/target",
            )
        )
        values.update(
            _named_distribution_metrics(
                prediction_distribution,
                metric_prefix=f"model/{split_name}/distribution/harmonic_relation/prediction",
            )
        )

    return values


def _named_distribution_metrics(distribution: tuple[float, ...] | None, *, metric_prefix: str) -> dict[str, float]:
    if distribution is None:
        return {}

    return {
        f"{metric_prefix}/{relation_id.name.lower()}": distribution[int(relation_id)]
        for relation_id in HarmonicRelationId
        if int(relation_id) < len(distribution)
    }


_EPOCH_METRIC_NAME_MAP: Final[dict[str, str]] = {
    "train_loss": "model/train/mean/loss",
    "train_perplexity": "model/train/mean/perplexity",
    "train_token_accuracy": "model/train/rate/token_accuracy",
    "train_token_kind_accuracy": "model/train/rate/token_kind_accuracy",
    "train_event_kind_loss": "model/train/mean/event_kind_loss",
    "train_duration_loss": "model/train/mean/duration_loss",
    "train_degree_loss": "model/train/mean/degree_loss",
    "train_accidental_loss": "model/train/mean/accidental_loss",
    "train_octave_offset_loss": "model/train/mean/octave_offset_loss",
    "train_hand_loss": "model/train/mean/hand_loss",
    "train_duration_accuracy": "model/train/rate/duration_accuracy",
    "train_degree_accuracy": "model/train/rate/degree_accuracy",
    "train_accidental_accuracy": "model/train/rate/accidental_accuracy",
    "train_octave_offset_accuracy": "model/train/rate/octave_offset_accuracy",
    "train_hand_accuracy": "model/train/rate/hand_accuracy",
    "train_musical_auxiliary_loss": "model/train/mean/musical_auxiliary_loss",
    "train_note_density_loss": "model/train/mean/note_density_loss",
    "train_note_density_accuracy": "model/train/rate/note_density_accuracy",
    "train_rhythmic_diversity_loss": "model/train/mean/rhythmic_diversity_loss",
    "train_rhythmic_diversity_accuracy": "model/train/rate/rhythmic_diversity_accuracy",
    "train_voice_independence_loss": "model/train/mean/voice_independence_loss",
    "train_voice_independence_accuracy": "model/train/rate/voice_independence_accuracy",
    "train_uses_accidentals_loss": "model/train/mean/uses_accidentals_loss",
    "train_uses_accidentals_accuracy": "model/train/rate/uses_accidentals_accuracy",
    "train_dotted_duration_loss": "model/train/mean/dotted_duration_loss",
    "train_dotted_duration_accuracy": "model/train/rate/dotted_duration_accuracy",
    "train_hand_span_loss": "model/train/mean/hand_span_loss",
    "train_hand_span_accuracy": "model/train/rate/hand_span_accuracy",
    "train_bar_note_density_loss": "model/train/mean/bar_note_density_loss",
    "train_bar_note_density_accuracy": "model/train/rate/bar_note_density_accuracy",
    "train_bar_rhythmic_diversity_loss": "model/train/mean/bar_rhythmic_diversity_loss",
    "train_bar_rhythmic_diversity_accuracy": "model/train/rate/bar_rhythmic_diversity_accuracy",
    "train_bar_voice_independence_loss": "model/train/mean/bar_voice_independence_loss",
    "train_bar_voice_independence_accuracy": "model/train/rate/bar_voice_independence_accuracy",
    "train_bar_uses_accidentals_loss": "model/train/mean/bar_uses_accidentals_loss",
    "train_bar_uses_accidentals_accuracy": "model/train/rate/bar_uses_accidentals_accuracy",
    "train_bar_dotted_duration_loss": "model/train/mean/bar_dotted_duration_loss",
    "train_bar_dotted_duration_accuracy": "model/train/rate/bar_dotted_duration_accuracy",
    "train_bar_hand_span_loss": "model/train/mean/bar_hand_span_loss",
    "train_bar_hand_span_accuracy": "model/train/rate/bar_hand_span_accuracy",
    "train_harmonic_relation_loss": "model/train/mean/harmonic_relation_loss",
    "train_harmonic_relation_accuracy": "model/train/rate/harmonic_relation_accuracy",
    "train_harmonic_relation_macro_f1": "model/train/rate/harmonic_relation_macro_f1",
    "train_harmony_gate_mean": "model/train/mean/harmony_gate",
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
    "validation_event_kind_loss": "model/validation/mean/event_kind_loss",
    "validation_duration_loss": "model/validation/mean/duration_loss",
    "validation_degree_loss": "model/validation/mean/degree_loss",
    "validation_accidental_loss": "model/validation/mean/accidental_loss",
    "validation_octave_offset_loss": "model/validation/mean/octave_offset_loss",
    "validation_hand_loss": "model/validation/mean/hand_loss",
    "validation_duration_accuracy": "model/validation/rate/duration_accuracy",
    "validation_degree_accuracy": "model/validation/rate/degree_accuracy",
    "validation_accidental_accuracy": "model/validation/rate/accidental_accuracy",
    "validation_octave_offset_accuracy": "model/validation/rate/octave_offset_accuracy",
    "validation_hand_accuracy": "model/validation/rate/hand_accuracy",
    "validation_musical_auxiliary_loss": "model/validation/mean/musical_auxiliary_loss",
    "validation_note_density_loss": "model/validation/mean/note_density_loss",
    "validation_note_density_accuracy": "model/validation/rate/note_density_accuracy",
    "validation_rhythmic_diversity_loss": "model/validation/mean/rhythmic_diversity_loss",
    "validation_rhythmic_diversity_accuracy": "model/validation/rate/rhythmic_diversity_accuracy",
    "validation_voice_independence_loss": "model/validation/mean/voice_independence_loss",
    "validation_voice_independence_accuracy": "model/validation/rate/voice_independence_accuracy",
    "validation_uses_accidentals_loss": "model/validation/mean/uses_accidentals_loss",
    "validation_uses_accidentals_accuracy": "model/validation/rate/uses_accidentals_accuracy",
    "validation_dotted_duration_loss": "model/validation/mean/dotted_duration_loss",
    "validation_dotted_duration_accuracy": "model/validation/rate/dotted_duration_accuracy",
    "validation_hand_span_loss": "model/validation/mean/hand_span_loss",
    "validation_hand_span_accuracy": "model/validation/rate/hand_span_accuracy",
    "validation_bar_note_density_loss": "model/validation/mean/bar_note_density_loss",
    "validation_bar_note_density_accuracy": "model/validation/rate/bar_note_density_accuracy",
    "validation_bar_rhythmic_diversity_loss": "model/validation/mean/bar_rhythmic_diversity_loss",
    "validation_bar_rhythmic_diversity_accuracy": "model/validation/rate/bar_rhythmic_diversity_accuracy",
    "validation_bar_voice_independence_loss": "model/validation/mean/bar_voice_independence_loss",
    "validation_bar_voice_independence_accuracy": "model/validation/rate/bar_voice_independence_accuracy",
    "validation_bar_uses_accidentals_loss": "model/validation/mean/bar_uses_accidentals_loss",
    "validation_bar_uses_accidentals_accuracy": "model/validation/rate/bar_uses_accidentals_accuracy",
    "validation_bar_dotted_duration_loss": "model/validation/mean/bar_dotted_duration_loss",
    "validation_bar_dotted_duration_accuracy": "model/validation/rate/bar_dotted_duration_accuracy",
    "validation_bar_hand_span_loss": "model/validation/mean/bar_hand_span_loss",
    "validation_bar_hand_span_accuracy": "model/validation/rate/bar_hand_span_accuracy",
    "validation_harmonic_relation_loss": "model/validation/mean/harmonic_relation_loss",
    "validation_harmonic_relation_accuracy": "model/validation/rate/harmonic_relation_accuracy",
    "validation_harmonic_relation_macro_f1": "model/validation/rate/harmonic_relation_macro_f1",
    "validation_harmony_gate_mean": "model/validation/mean/harmony_gate",
    "validation_validity_penalty_loss": "model/validation/mean/validity_penalty_loss",
    "validation_invalid_probability_mass": "model/validation/mean/invalid_probability_mass",
    "validation_invalid_target_rate": "model/validation/rate/invalid_target",
}
