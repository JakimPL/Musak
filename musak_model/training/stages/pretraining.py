import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from musak_model.conditioning.structural.vocabulary import StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.data.config import SegmentationConfig
from musak_model.evaluation import GenerationSuiteEvaluator
from musak_model.evaluation.generation import GenerationEvaluator
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import ModelConfig, ModelOutputMode
from musak_model.model.output import ModelTrainingLogits
from musak_model.paths import CONDITIONING_CONFIG_PATH
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint import load_checkpoint, save_checkpoint
from musak_model.training.config import TrainingConfig
from musak_model.training.dataset.factorized import TokenAttributeTargetTensors
from musak_model.training.dataset.loaders import build_dataloaders
from musak_model.training.dataset.schema import TrainingBatch
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.schema import IngestionErrorRecord
from musak_model.training.ingestion.split import build_split
from musak_model.training.losses import (
    FactorizedEventLoss,
    MusicalAuxiliaryLoss,
    factorized_event_loss,
    musical_auxiliary_loss,
)
from musak_model.training.metrics import (
    BatchMetrics,
    EpochMetrics,
    EpochSplitMetrics,
    MetricsAccumulator,
    batch_metrics_from_logits,
    build_token_attribute_lookup,
    build_token_kind_ids,
    module_gradient_norm_metrics,
)
from musak_model.training.progress import log_split_summary, progress
from musak_model.training.stages.auxiliary_profiles import split_musical_auxiliary_profile_metrics
from musak_model.training.stages.figure_profiles import (
    load_generation_figure_profile_artifacts,
    split_figure_profile_metrics,
)
from musak_model.training.tracking import NoOpTrainingTracker, TrainingTracker, build_training_tracker
from musak_model.training.validity import TrainingValidityMaskBuilder

_LOGGER = logging.getLogger(__name__)


@dataclass
class EarlyStoppingState:
    best_validation_loss: float | None
    stale_epoch_count: int = 0


class TrainingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    metrics: list[EpochMetrics]
    best_checkpoint_path: Path | None
    latest_checkpoint_path: Path | None
    invalid_files: list[IngestionErrorRecord]


class PretrainingTrainer:
    def __init__(
        self,
        *,
        model: HierarchicalAutoregressiveModel,
        config: TrainingConfig,
        train_loader: DataLoader[TrainingBatch],
        validation_loader: DataLoader[TrainingBatch],
        tracker: TrainingTracker | None = None,
        show_progress: bool = False,
        token_kind_ids: Tensor | None = None,
        token_attribute_lookup: TokenAttributeTargetTensors | None = None,
        validity_mask_builder: TrainingValidityMaskBuilder | None = None,
        generation_evaluator: GenerationEvaluator | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._device = torch.device(config.runtime.device)
        self._optimizer = AdamW(
            self._model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        self._model.to(self._device)
        self._tracker = tracker or NoOpTrainingTracker()
        self._show_progress = show_progress
        self._token_kind_ids = token_kind_ids.to(self._device) if token_kind_ids is not None else None
        self._token_attribute_lookup = (
            token_attribute_lookup.to(self._device) if token_attribute_lookup is not None else None
        )
        if config.conditioning.use_validity_penalty and validity_mask_builder is None:
            raise ValueError("validity_mask_builder is required when use_validity_penalty is true")

        if self._model.output_mode != config.event_objective.mode:
            raise ValueError(
                f"model output mode {self._model.output_mode.value!r} does not match "
                f"training objective mode {config.event_objective.mode.value!r}"
            )

        self._validity_mask_builder = validity_mask_builder
        self._generation_evaluator = generation_evaluator

    def train(self, *, invalid_files: list[IngestionErrorRecord] | None = None) -> TrainingResult:
        if len(self._train_loader) == 0:
            raise ValueError("training loader is empty")

        self._log_training_shape()
        start_epoch, best_validation_loss = self._resume_training_state()
        metrics: list[EpochMetrics] = []
        best_checkpoint_path: Path | None = None
        latest_checkpoint_path = self._config.checkpoints.checkpoint_directory / "latest.pt"
        early_stopping_state = EarlyStoppingState(best_validation_loss=best_validation_loss)

        for epoch in range(start_epoch, self._config.optimization.epochs):
            metric = self._run_epoch(epoch=epoch)
            metrics.append(metric)
            best_validation_loss, best_checkpoint_path = self._save_checkpoints(
                epoch=epoch,
                metric=metric,
                best_validation_loss=best_validation_loss,
                best_checkpoint_path=best_checkpoint_path,
                latest_checkpoint_path=latest_checkpoint_path,
            )
            if self._should_stop_early(metric=metric, state=early_stopping_state):
                break

        return self._finish_training(
            metrics=metrics,
            best_checkpoint_path=best_checkpoint_path,
            latest_checkpoint_path=latest_checkpoint_path,
            invalid_files=invalid_files,
        )

    def _log_training_shape(self) -> None:
        _LOGGER.info("Training batches per epoch: %s", len(self._train_loader))
        _LOGGER.info("Validation batches per epoch: %s", len(self._validation_loader))
        early_stopping = self._config.early_stopping
        if early_stopping.enabled:
            _LOGGER.info(
                "Early stopping enabled: monitor=validation_loss patience_epochs=%s min_delta=%s",
                early_stopping.patience_epochs,
                early_stopping.min_delta,
            )

    def _resume_training_state(self) -> tuple[int, float | None]:
        if self._config.checkpoints.resume_checkpoint is None:
            return 0, None

        _LOGGER.info("Resuming from checkpoint: %s", self._config.checkpoints.resume_checkpoint)
        return load_checkpoint(
            self._config.checkpoints.resume_checkpoint,
            model=self._model,
            optimizer=self._optimizer,
            device=self._device,
        )

    def _run_epoch(self, *, epoch: int) -> EpochMetrics:
        _LOGGER.info("Epoch %s/%s started", epoch + 1, self._config.optimization.epochs)
        metric = self._epoch_metrics(
            epoch=epoch,
            train_metrics=self._train_epoch(epoch=epoch),
            validation_metrics=self._validate_epoch(epoch=epoch),
        )
        self._tracker.log_epoch(metrics=metric)
        self._log_generation_evaluation(epoch=epoch)
        self._log_epoch_result(metric)
        return metric

    def _epoch_metrics(
        self,
        *,
        epoch: int,
        train_metrics: EpochSplitMetrics,
        validation_metrics: EpochSplitMetrics | None,
    ) -> EpochMetrics:
        return EpochMetrics(
            epoch=epoch,
            train_loss=train_metrics.loss,
            train_perplexity=train_metrics.perplexity,
            train_token_accuracy=train_metrics.token_accuracy,
            train_token_kind_accuracy=train_metrics.token_kind_accuracy,
            train_event_kind_loss=train_metrics.event_kind_loss,
            train_duration_loss=train_metrics.duration_loss,
            train_degree_loss=train_metrics.degree_loss,
            train_accidental_loss=train_metrics.accidental_loss,
            train_octave_offset_loss=train_metrics.octave_offset_loss,
            train_hand_loss=train_metrics.hand_loss,
            train_duration_accuracy=train_metrics.duration_accuracy,
            train_degree_accuracy=train_metrics.degree_accuracy,
            train_accidental_accuracy=train_metrics.accidental_accuracy,
            train_octave_offset_accuracy=train_metrics.octave_offset_accuracy,
            train_hand_accuracy=train_metrics.hand_accuracy,
            train_musical_auxiliary_loss=train_metrics.musical_auxiliary_loss,
            train_note_density_loss=train_metrics.note_density_loss,
            train_note_density_accuracy=train_metrics.note_density_accuracy,
            train_rhythmic_diversity_loss=train_metrics.rhythmic_diversity_loss,
            train_rhythmic_diversity_accuracy=train_metrics.rhythmic_diversity_accuracy,
            train_voice_independence_loss=train_metrics.voice_independence_loss,
            train_voice_independence_accuracy=train_metrics.voice_independence_accuracy,
            train_uses_accidentals_loss=train_metrics.uses_accidentals_loss,
            train_uses_accidentals_accuracy=train_metrics.uses_accidentals_accuracy,
            train_dotted_duration_loss=train_metrics.dotted_duration_loss,
            train_dotted_duration_accuracy=train_metrics.dotted_duration_accuracy,
            train_hand_span_loss=train_metrics.hand_span_loss,
            train_hand_span_accuracy=train_metrics.hand_span_accuracy,
            train_bar_note_density_loss=train_metrics.bar_note_density_loss,
            train_bar_note_density_accuracy=train_metrics.bar_note_density_accuracy,
            train_bar_rhythmic_diversity_loss=train_metrics.bar_rhythmic_diversity_loss,
            train_bar_rhythmic_diversity_accuracy=train_metrics.bar_rhythmic_diversity_accuracy,
            train_bar_voice_independence_loss=train_metrics.bar_voice_independence_loss,
            train_bar_voice_independence_accuracy=train_metrics.bar_voice_independence_accuracy,
            train_bar_uses_accidentals_loss=train_metrics.bar_uses_accidentals_loss,
            train_bar_uses_accidentals_accuracy=train_metrics.bar_uses_accidentals_accuracy,
            train_bar_dotted_duration_loss=train_metrics.bar_dotted_duration_loss,
            train_bar_dotted_duration_accuracy=train_metrics.bar_dotted_duration_accuracy,
            train_bar_hand_span_loss=train_metrics.bar_hand_span_loss,
            train_bar_hand_span_accuracy=train_metrics.bar_hand_span_accuracy,
            train_validity_penalty_loss=train_metrics.validity_penalty_loss,
            train_invalid_probability_mass=train_metrics.invalid_probability_mass,
            train_invalid_target_rate=train_metrics.invalid_target_rate,
            train_cnn_gradient_norm=train_metrics.cnn_gradient_norm,
            train_gru_gradient_norm=train_metrics.gru_gradient_norm,
            train_transformer_gradient_norm=train_metrics.transformer_gradient_norm,
            validation_loss=validation_metrics.loss if validation_metrics is not None else None,
            validation_perplexity=validation_metrics.perplexity if validation_metrics is not None else None,
            validation_token_accuracy=validation_metrics.token_accuracy if validation_metrics is not None else None,
            validation_token_kind_accuracy=(
                validation_metrics.token_kind_accuracy if validation_metrics is not None else None
            ),
            validation_event_kind_loss=validation_metrics.event_kind_loss if validation_metrics is not None else None,
            validation_duration_loss=validation_metrics.duration_loss if validation_metrics is not None else None,
            validation_degree_loss=validation_metrics.degree_loss if validation_metrics is not None else None,
            validation_accidental_loss=validation_metrics.accidental_loss if validation_metrics is not None else None,
            validation_octave_offset_loss=(
                validation_metrics.octave_offset_loss if validation_metrics is not None else None
            ),
            validation_hand_loss=validation_metrics.hand_loss if validation_metrics is not None else None,
            validation_duration_accuracy=(
                validation_metrics.duration_accuracy if validation_metrics is not None else None
            ),
            validation_degree_accuracy=validation_metrics.degree_accuracy if validation_metrics is not None else None,
            validation_accidental_accuracy=(
                validation_metrics.accidental_accuracy if validation_metrics is not None else None
            ),
            validation_octave_offset_accuracy=(
                validation_metrics.octave_offset_accuracy if validation_metrics is not None else None
            ),
            validation_hand_accuracy=validation_metrics.hand_accuracy if validation_metrics is not None else None,
            validation_musical_auxiliary_loss=(
                validation_metrics.musical_auxiliary_loss if validation_metrics is not None else None
            ),
            validation_note_density_loss=(
                validation_metrics.note_density_loss if validation_metrics is not None else None
            ),
            validation_note_density_accuracy=(
                validation_metrics.note_density_accuracy if validation_metrics is not None else None
            ),
            validation_rhythmic_diversity_loss=(
                validation_metrics.rhythmic_diversity_loss if validation_metrics is not None else None
            ),
            validation_rhythmic_diversity_accuracy=(
                validation_metrics.rhythmic_diversity_accuracy if validation_metrics is not None else None
            ),
            validation_voice_independence_loss=(
                validation_metrics.voice_independence_loss if validation_metrics is not None else None
            ),
            validation_voice_independence_accuracy=(
                validation_metrics.voice_independence_accuracy if validation_metrics is not None else None
            ),
            validation_uses_accidentals_loss=(
                validation_metrics.uses_accidentals_loss if validation_metrics is not None else None
            ),
            validation_uses_accidentals_accuracy=(
                validation_metrics.uses_accidentals_accuracy if validation_metrics is not None else None
            ),
            validation_dotted_duration_loss=(
                validation_metrics.dotted_duration_loss if validation_metrics is not None else None
            ),
            validation_dotted_duration_accuracy=(
                validation_metrics.dotted_duration_accuracy if validation_metrics is not None else None
            ),
            validation_hand_span_loss=validation_metrics.hand_span_loss if validation_metrics is not None else None,
            validation_hand_span_accuracy=(
                validation_metrics.hand_span_accuracy if validation_metrics is not None else None
            ),
            validation_bar_note_density_loss=(
                validation_metrics.bar_note_density_loss if validation_metrics is not None else None
            ),
            validation_bar_note_density_accuracy=(
                validation_metrics.bar_note_density_accuracy if validation_metrics is not None else None
            ),
            validation_bar_rhythmic_diversity_loss=(
                validation_metrics.bar_rhythmic_diversity_loss if validation_metrics is not None else None
            ),
            validation_bar_rhythmic_diversity_accuracy=(
                validation_metrics.bar_rhythmic_diversity_accuracy if validation_metrics is not None else None
            ),
            validation_bar_voice_independence_loss=(
                validation_metrics.bar_voice_independence_loss if validation_metrics is not None else None
            ),
            validation_bar_voice_independence_accuracy=(
                validation_metrics.bar_voice_independence_accuracy if validation_metrics is not None else None
            ),
            validation_bar_uses_accidentals_loss=(
                validation_metrics.bar_uses_accidentals_loss if validation_metrics is not None else None
            ),
            validation_bar_uses_accidentals_accuracy=(
                validation_metrics.bar_uses_accidentals_accuracy if validation_metrics is not None else None
            ),
            validation_bar_dotted_duration_loss=(
                validation_metrics.bar_dotted_duration_loss if validation_metrics is not None else None
            ),
            validation_bar_dotted_duration_accuracy=(
                validation_metrics.bar_dotted_duration_accuracy if validation_metrics is not None else None
            ),
            validation_bar_hand_span_loss=(
                validation_metrics.bar_hand_span_loss if validation_metrics is not None else None
            ),
            validation_bar_hand_span_accuracy=(
                validation_metrics.bar_hand_span_accuracy if validation_metrics is not None else None
            ),
            validation_validity_penalty_loss=(
                validation_metrics.validity_penalty_loss if validation_metrics is not None else None
            ),
            validation_invalid_probability_mass=(
                validation_metrics.invalid_probability_mass if validation_metrics is not None else None
            ),
            validation_invalid_target_rate=(
                validation_metrics.invalid_target_rate if validation_metrics is not None else None
            ),
        )

    def _log_epoch_result(self, metric: EpochMetrics) -> None:
        _LOGGER.info(
            (
                "Epoch %s/%s finished: train_loss=%.6f train_perplexity=%.6f "
                "train_token_accuracy=%.6f train_token_kind_accuracy=%s train_event_kind_loss=%s "
                "train_duration_loss=%s train_degree_loss=%s train_accidental_loss=%s "
                "train_octave_offset_loss=%s train_hand_loss=%s train_duration_accuracy=%s "
                "train_degree_accuracy=%s train_accidental_accuracy=%s train_octave_offset_accuracy=%s "
                "train_hand_accuracy=%s train_validity_penalty_loss=%s "
                "train_invalid_probability_mass=%s train_invalid_target_rate=%s train_cnn_gradient_norm=%s "
                "train_gru_gradient_norm=%s train_transformer_gradient_norm=%s validation_loss=%s "
                "validation_perplexity=%s validation_token_accuracy=%s validation_token_kind_accuracy=%s "
                "validation_event_kind_loss=%s validation_duration_loss=%s validation_degree_loss=%s "
                "validation_accidental_loss=%s validation_octave_offset_loss=%s validation_hand_loss=%s "
                "validation_duration_accuracy=%s validation_degree_accuracy=%s validation_accidental_accuracy=%s "
                "validation_octave_offset_accuracy=%s validation_hand_accuracy=%s "
                "validation_validity_penalty_loss=%s validation_invalid_probability_mass=%s "
                "validation_invalid_target_rate=%s"
            ),
            metric.epoch + 1,
            self._config.optimization.epochs,
            metric.train_loss,
            metric.train_perplexity,
            metric.train_token_accuracy,
            metric.train_token_kind_accuracy,
            metric.train_event_kind_loss,
            metric.train_duration_loss,
            metric.train_degree_loss,
            metric.train_accidental_loss,
            metric.train_octave_offset_loss,
            metric.train_hand_loss,
            metric.train_duration_accuracy,
            metric.train_degree_accuracy,
            metric.train_accidental_accuracy,
            metric.train_octave_offset_accuracy,
            metric.train_hand_accuracy,
            metric.train_validity_penalty_loss,
            metric.train_invalid_probability_mass,
            metric.train_invalid_target_rate,
            metric.train_cnn_gradient_norm,
            metric.train_gru_gradient_norm,
            metric.train_transformer_gradient_norm,
            metric.validation_loss,
            metric.validation_perplexity,
            metric.validation_token_accuracy,
            metric.validation_token_kind_accuracy,
            metric.validation_event_kind_loss,
            metric.validation_duration_loss,
            metric.validation_degree_loss,
            metric.validation_accidental_loss,
            metric.validation_octave_offset_loss,
            metric.validation_hand_loss,
            metric.validation_duration_accuracy,
            metric.validation_degree_accuracy,
            metric.validation_accidental_accuracy,
            metric.validation_octave_offset_accuracy,
            metric.validation_hand_accuracy,
            metric.validation_validity_penalty_loss,
            metric.validation_invalid_probability_mass,
            metric.validation_invalid_target_rate,
        )

    def _save_checkpoints(
        self,
        *,
        epoch: int,
        metric: EpochMetrics,
        best_validation_loss: float | None,
        best_checkpoint_path: Path | None,
        latest_checkpoint_path: Path,
    ) -> tuple[float | None, Path | None]:
        best_validation_loss, best_checkpoint_path = self._save_best_checkpoint(
            epoch=epoch,
            metric=metric,
            best_validation_loss=best_validation_loss,
            best_checkpoint_path=best_checkpoint_path,
        )
        self._save_checkpoint(latest_checkpoint_path, epoch=epoch, best_validation_loss=best_validation_loss)
        _LOGGER.info("Saved latest checkpoint: %s", latest_checkpoint_path)
        self._save_epoch_checkpoint(epoch=epoch, best_validation_loss=best_validation_loss)
        return best_validation_loss, best_checkpoint_path

    def _save_checkpoint(self, path: Path, *, epoch: int, best_validation_loss: float | None) -> None:
        save_checkpoint(
            path,
            model=self._model,
            optimizer=self._optimizer,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
        )

    def _save_epoch_checkpoint(self, *, epoch: int, best_validation_loss: float | None) -> None:
        if not self._config.checkpoints.save_all_epochs:
            return

        epoch_checkpoint_path = self._config.checkpoints.checkpoint_directory / f"epoch_{epoch:04d}.pt"
        self._save_checkpoint(epoch_checkpoint_path, epoch=epoch, best_validation_loss=best_validation_loss)
        _LOGGER.info("Saved epoch checkpoint: %s", epoch_checkpoint_path)

    def _save_best_checkpoint(
        self,
        *,
        epoch: int,
        metric: EpochMetrics,
        best_validation_loss: float | None,
        best_checkpoint_path: Path | None,
    ) -> tuple[float | None, Path | None]:
        score = metric.validation_loss if metric.validation_loss is not None else metric.train_loss
        if best_validation_loss is not None and score >= best_validation_loss:
            return best_validation_loss, best_checkpoint_path

        best_validation_loss = score
        best_checkpoint_path = self._config.checkpoints.checkpoint_directory / "best.pt"
        self._save_checkpoint(best_checkpoint_path, epoch=epoch, best_validation_loss=best_validation_loss)
        _LOGGER.info("Saved best checkpoint: %s", best_checkpoint_path)
        return best_validation_loss, best_checkpoint_path

    def _should_stop_early(self, *, metric: EpochMetrics, state: EarlyStoppingState) -> bool:
        early_stopping = self._config.early_stopping
        if not early_stopping.enabled:
            return False

        if metric.validation_loss is None:
            return False

        if _validation_loss_improved(
            validation_loss=metric.validation_loss,
            best_validation_loss=state.best_validation_loss,
            min_delta=early_stopping.min_delta,
        ):
            state.best_validation_loss = metric.validation_loss
            state.stale_epoch_count = 0
            return False

        state.stale_epoch_count += 1
        _LOGGER.info(
            "Early stopping patience: stale_epochs=%s/%s best_validation_loss=%s current_validation_loss=%s",
            state.stale_epoch_count,
            early_stopping.patience_epochs,
            state.best_validation_loss,
            metric.validation_loss,
        )
        if state.stale_epoch_count < early_stopping.patience_epochs:
            return False

        _LOGGER.info("Stopping early after epoch %s", metric.epoch + 1)
        return True

    def _finish_training(
        self,
        *,
        metrics: list[EpochMetrics],
        best_checkpoint_path: Path | None,
        latest_checkpoint_path: Path,
        invalid_files: list[IngestionErrorRecord] | None,
    ) -> TrainingResult:
        result = TrainingResult(
            metrics=metrics,
            best_checkpoint_path=best_checkpoint_path,
            latest_checkpoint_path=latest_checkpoint_path,
            invalid_files=invalid_files or [],
        )
        self._tracker.log_checkpoints(
            latest_checkpoint_path=result.latest_checkpoint_path,
            best_checkpoint_path=result.best_checkpoint_path,
        )
        self._tracker.log_invalid_files(invalid_files=result.invalid_files)
        return result

    def _log_generation_evaluation(self, *, epoch: int) -> None:
        evaluation_config = self._config.generation_evaluation
        if not evaluation_config.enabled:
            return

        if self._generation_evaluator is None:
            return

        if (epoch + 1) % evaluation_config.every_epochs != 0:
            return

        _LOGGER.info("Running generation evaluation for epoch %s", epoch + 1)
        result = self._generation_evaluator.evaluate_result(self._model, device=self._device)
        self._tracker.log_generation_evaluation(metrics=result.metrics, epoch=epoch)
        with TemporaryDirectory(prefix="musak-generation-evaluation-") as temporary_directory:
            artifact_directory = Path(temporary_directory)
            self._generation_evaluator.write_artifacts(result, output_directory=artifact_directory)
            self._tracker.log_generation_artifacts(artifact_directory=artifact_directory, epoch=epoch)
        _LOGGER.info("Logged %s generation evaluation metric(s)", len(result.metrics))

    def _train_epoch(self, *, epoch: int) -> EpochSplitMetrics:
        self._model.train()
        accumulator = MetricsAccumulator()

        for batch in progress(
            self._train_loader,
            description=f"Training epoch {epoch + 1}",
            unit="batch",
            enabled=self._show_progress,
            total=len(self._train_loader),
        ):
            batch = _move_batch_to_device(batch, device=self._device)
            self._optimizer.zero_grad(set_to_none=True)
            loss, batch_metrics = self._loss_for_batch(batch)
            loss.backward()  # type: ignore[no-untyped-call]
            batch_metrics = batch_metrics.model_copy(update=module_gradient_norm_metrics(self._model))
            self._optimizer.step()
            accumulator.add(batch_metrics)

        return accumulator.to_epoch_split_metrics()

    def _validate_epoch(self, *, epoch: int) -> EpochSplitMetrics | None:
        if len(self._validation_loader) == 0:
            return None

        self._model.eval()
        accumulator = MetricsAccumulator()
        with torch.no_grad():
            for batch in progress(
                self._validation_loader,
                description=f"Validation epoch {epoch + 1}",
                unit="batch",
                enabled=self._show_progress,
                total=len(self._validation_loader),
            ):
                batch = _move_batch_to_device(batch, device=self._device)
                _, batch_metrics = self._loss_for_batch(batch)
                accumulator.add(batch_metrics)

        return accumulator.to_epoch_split_metrics()

    def _loss_for_batch(self, batch: TrainingBatch) -> tuple[Tensor, BatchMetrics]:
        valid_mask = ~batch.token_padding_mask.reshape(-1)
        model_logits = self._model_training_logits(batch)
        logits, loss, factorized_loss = self._event_logits_and_loss(
            batch,
            model_logits=model_logits,
            valid_mask=valid_mask,
        )
        auxiliary_loss = self._musical_auxiliary_loss(model_logits, batch=batch)
        if auxiliary_loss is not None:
            loss = loss + self._config.musical_auxiliary_objective.weight * auxiliary_loss.loss

        log_probabilities = nn.functional.log_softmax(logits, dim=-1)
        batch_metrics = batch_metrics_from_logits(
            logits,
            target_token_ids=batch.target_token_ids,
            token_padding_mask=batch.token_padding_mask,
            loss=loss,
            token_kind_ids=self._token_kind_ids,
            token_attribute_lookup=self._token_attribute_lookup,
        )
        batch_metrics = self._add_factorized_loss_metrics(batch_metrics, factorized_loss=factorized_loss)
        batch_metrics = self._add_musical_auxiliary_loss_metrics(batch_metrics, auxiliary_loss=auxiliary_loss)
        if self._config.conditioning.use_validity_penalty:
            validity_metrics = self._validity_penalty_metrics(
                log_probabilities,
                batch=batch,
                valid_mask=valid_mask.reshape(batch.token_padding_mask.shape),
            )
            loss = loss + self._config.conditioning.validity_penalty_weight * validity_metrics["penalty_loss"]
            batch_metrics = batch_metrics.model_copy(
                update={
                    "loss": float(loss.detach().item()),
                    "validity_penalty_loss": float(validity_metrics["penalty_loss"].detach().item()),
                    "invalid_probability_mass": float(validity_metrics["invalid_mass"].detach().item()),
                    "invalid_target_count": int(validity_metrics["invalid_target_count"].detach().item()),
                    "validity_penalty_token_count": int(validity_metrics["penalty_token_count"].detach().item()),
                }
            )
        return loss, batch_metrics

    def _model_training_logits(self, batch: TrainingBatch) -> ModelTrainingLogits:
        return self._model.training_logits(
            batch.input_token_ids,
            bar_positions=batch.bar_positions,
            bar_relative_ticks=batch.bar_relative_ticks,
            bar_duration_ticks=batch.bar_duration_ticks,
            active_hand_ids=batch.active_hand_ids,
            target_bar_positions=batch.target_bar_positions,
            bar_counts=batch.bar_counts,
            difficulty_ids=batch.difficulty_ids if self._config.conditioning.use_difficulty else None,
            scale_type_ids=batch.conditioning_scale_type_ids if self._config.conditioning.use_scale_type else None,
            time_signature_ids=(
                batch.conditioning_time_signature_ids if self._config.conditioning.use_time_signature else None
            ),
            structural_control_ids=(
                batch.structural_control_ids if self._config.conditioning.use_structural_conditioning else None
            ),
            token_padding_mask=batch.token_padding_mask,
        )

    def _event_logits_and_loss(
        self,
        batch: TrainingBatch,
        *,
        model_logits: ModelTrainingLogits,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, FactorizedEventLoss | None]:
        match self._config.event_objective.mode:
            case ModelOutputMode.FLAT:
                return self._flat_logits_and_loss(batch, model_logits=model_logits, valid_mask=valid_mask)
            case ModelOutputMode.FACTORIZED:
                return self._factorized_logits_and_loss(batch, model_logits=model_logits)

    def _flat_logits_and_loss(
        self,
        batch: TrainingBatch,
        *,
        model_logits: ModelTrainingLogits,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, None]:
        logits = model_logits.flat_logits
        log_probabilities = nn.functional.log_softmax(logits, dim=-1)
        flat_loss = (
            -log_probabilities.gather(dim=-1, index=batch.target_token_ids.unsqueeze(-1)).squeeze(-1).reshape(-1)
        )
        cross_entropy_loss = flat_loss[valid_mask].sum() / int(valid_mask.sum().item())
        return logits, cross_entropy_loss, None

    def _factorized_logits_and_loss(
        self,
        batch: TrainingBatch,
        *,
        model_logits: ModelTrainingLogits,
    ) -> tuple[Tensor, Tensor, FactorizedEventLoss]:
        factorized_logits = model_logits.factorized_logits
        if factorized_logits is None:
            raise ValueError("factorized training requires factorized logits")

        factorized_loss = factorized_event_loss(
            factorized_logits,
            targets=batch.target_token_attributes,
            config=self._config.event_objective,
        )
        return model_logits.flat_logits, factorized_loss.loss, factorized_loss

    def _musical_auxiliary_loss(
        self,
        model_logits: ModelTrainingLogits,
        *,
        batch: TrainingBatch,
    ) -> MusicalAuxiliaryLoss | None:
        if not self._config.musical_auxiliary_objective.enabled:
            return None

        return musical_auxiliary_loss(
            model_logits.musical_auxiliary_logits,
            targets=batch.musical_auxiliary_targets,
            config=self._config.musical_auxiliary_objective,
        )

    def _add_factorized_loss_metrics(
        self,
        batch_metrics: BatchMetrics,
        *,
        factorized_loss: FactorizedEventLoss | None,
    ) -> BatchMetrics:
        if factorized_loss is None:
            return batch_metrics

        return batch_metrics.model_copy(
            update={
                "event_kind_loss": float(factorized_loss.kind_loss.detach().item()),
                "event_kind_loss_target_count": factorized_loss.kind_target_count,
                "duration_loss": float(factorized_loss.duration_loss.detach().item()),
                "duration_loss_target_count": factorized_loss.duration_target_count,
                "degree_loss": float(factorized_loss.degree_loss.detach().item()),
                "degree_loss_target_count": factorized_loss.degree_target_count,
                "accidental_loss": float(factorized_loss.accidental_loss.detach().item()),
                "accidental_loss_target_count": factorized_loss.accidental_target_count,
                "octave_offset_loss": float(factorized_loss.octave_offset_loss.detach().item()),
                "octave_offset_loss_target_count": factorized_loss.octave_offset_target_count,
                "hand_loss": float(factorized_loss.hand_loss.detach().item()),
                "hand_loss_target_count": factorized_loss.hand_target_count,
            }
        )

    def _add_musical_auxiliary_loss_metrics(
        self,
        batch_metrics: BatchMetrics,
        *,
        auxiliary_loss: MusicalAuxiliaryLoss | None,
    ) -> BatchMetrics:
        if auxiliary_loss is None:
            return batch_metrics

        total_target_count = (
            auxiliary_loss.note_density_target_count
            + auxiliary_loss.rhythmic_diversity_target_count
            + auxiliary_loss.voice_independence_target_count
            + auxiliary_loss.uses_accidentals_target_count
            + auxiliary_loss.dotted_duration_target_count
            + auxiliary_loss.hand_span_target_count
            + auxiliary_loss.bar_note_density_target_count
            + auxiliary_loss.bar_rhythmic_diversity_target_count
            + auxiliary_loss.bar_voice_independence_target_count
            + auxiliary_loss.bar_uses_accidentals_target_count
            + auxiliary_loss.bar_dotted_duration_target_count
            + auxiliary_loss.bar_hand_span_target_count
        )
        return batch_metrics.model_copy(
            update={
                "musical_auxiliary_loss": float(auxiliary_loss.loss.detach().item()),
                "musical_auxiliary_target_count": total_target_count,
                "note_density_loss": float(auxiliary_loss.note_density_loss.detach().item()),
                "note_density_match_count": auxiliary_loss.note_density_match_count,
                "note_density_target_count": auxiliary_loss.note_density_target_count,
                "rhythmic_diversity_loss": float(auxiliary_loss.rhythmic_diversity_loss.detach().item()),
                "rhythmic_diversity_match_count": auxiliary_loss.rhythmic_diversity_match_count,
                "rhythmic_diversity_target_count": auxiliary_loss.rhythmic_diversity_target_count,
                "voice_independence_loss": float(auxiliary_loss.voice_independence_loss.detach().item()),
                "voice_independence_match_count": auxiliary_loss.voice_independence_match_count,
                "voice_independence_target_count": auxiliary_loss.voice_independence_target_count,
                "uses_accidentals_loss": float(auxiliary_loss.uses_accidentals_loss.detach().item()),
                "uses_accidentals_match_count": auxiliary_loss.uses_accidentals_match_count,
                "uses_accidentals_target_count": auxiliary_loss.uses_accidentals_target_count,
                "dotted_duration_loss": float(auxiliary_loss.dotted_duration_loss.detach().item()),
                "dotted_duration_match_count": auxiliary_loss.dotted_duration_match_count,
                "dotted_duration_target_count": auxiliary_loss.dotted_duration_target_count,
                "hand_span_loss": float(auxiliary_loss.hand_span_loss.detach().item()),
                "hand_span_match_count": auxiliary_loss.hand_span_match_count,
                "hand_span_target_count": auxiliary_loss.hand_span_target_count,
                "bar_note_density_loss": float(auxiliary_loss.bar_note_density_loss.detach().item()),
                "bar_note_density_match_count": auxiliary_loss.bar_note_density_match_count,
                "bar_note_density_target_count": auxiliary_loss.bar_note_density_target_count,
                "bar_rhythmic_diversity_loss": float(auxiliary_loss.bar_rhythmic_diversity_loss.detach().item()),
                "bar_rhythmic_diversity_match_count": auxiliary_loss.bar_rhythmic_diversity_match_count,
                "bar_rhythmic_diversity_target_count": auxiliary_loss.bar_rhythmic_diversity_target_count,
                "bar_voice_independence_loss": float(auxiliary_loss.bar_voice_independence_loss.detach().item()),
                "bar_voice_independence_match_count": auxiliary_loss.bar_voice_independence_match_count,
                "bar_voice_independence_target_count": auxiliary_loss.bar_voice_independence_target_count,
                "bar_uses_accidentals_loss": float(auxiliary_loss.bar_uses_accidentals_loss.detach().item()),
                "bar_uses_accidentals_match_count": auxiliary_loss.bar_uses_accidentals_match_count,
                "bar_uses_accidentals_target_count": auxiliary_loss.bar_uses_accidentals_target_count,
                "bar_dotted_duration_loss": float(auxiliary_loss.bar_dotted_duration_loss.detach().item()),
                "bar_dotted_duration_match_count": auxiliary_loss.bar_dotted_duration_match_count,
                "bar_dotted_duration_target_count": auxiliary_loss.bar_dotted_duration_target_count,
                "bar_hand_span_loss": float(auxiliary_loss.bar_hand_span_loss.detach().item()),
                "bar_hand_span_match_count": auxiliary_loss.bar_hand_span_match_count,
                "bar_hand_span_target_count": auxiliary_loss.bar_hand_span_target_count,
            }
        )

    def _validity_penalty_metrics(
        self,
        log_probabilities: Tensor,
        *,
        batch: TrainingBatch,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        if self._validity_mask_builder is None:
            raise ValueError("validity_mask_builder is required")

        masks = self._validity_mask_builder.masks_for_batch(batch, device=log_probabilities.device)
        target_is_invalid = masks.invalid_target_mask & valid_mask
        penalty_mask = valid_mask & ~target_is_invalid
        penalty_token_count = penalty_mask.sum()
        if int(penalty_token_count.item()) == 0:
            zero = log_probabilities.sum() * 0.0
            return {
                "penalty_loss": zero,
                "invalid_mass": zero,
                "invalid_target_count": target_is_invalid.sum(),
                "penalty_token_count": penalty_token_count,
            }

        invalid_log_probabilities = log_probabilities.masked_fill(~masks.invalid_token_mask, float("-inf"))
        invalid_mass = torch.logsumexp(invalid_log_probabilities, dim=-1).exp()
        penalty_loss = invalid_mass[penalty_mask].mean()
        return {
            "penalty_loss": penalty_loss,
            "invalid_mass": penalty_loss.detach(),
            "invalid_target_count": target_is_invalid.sum(),
            "penalty_token_count": penalty_token_count,
        }


def pretrain(
    source_directory: Path,
    *,
    ingestion_config: IngestionConfig,
    segmentation_config: SegmentationConfig,
    training_config: TrainingConfig,
    tokenization_config: TokenizationConfig,
    model_config: ModelConfig | None = None,
    conditioning_config_path: Path = CONDITIONING_CONFIG_PATH,
    show_progress: bool = False,
    allow_raw_fallback: bool = True,
) -> TrainingResult:
    _LOGGER.info("Building train/validation split")
    split = build_split(
        source_directory,
        config=ingestion_config,
        segmentation=segmentation_config,
        tokenization_config=tokenization_config,
        allow_raw_fallback=allow_raw_fallback,
        show_progress=show_progress,
    )
    log_split_summary(split)
    vocabulary = TokenVocabulary(DurationVocabulary(tokenization_config))
    resolved_model_config = model_config or ModelConfig.load(
        vocabulary_size=vocabulary.vocabulary_size,
        duration_vocabulary_size=vocabulary.duration_vocabulary.vocabulary_size(),
        output_mode=training_config.event_objective.mode,
        musical_auxiliary_targets=training_config.musical_auxiliary_targets,
        conditioning_config_path=conditioning_config_path,
    )
    _LOGGER.info("Model vocabulary size: %s", resolved_model_config.vocabulary_size)
    time_signature_vocabulary = TimeSignatureVocabulary(resolved_model_config.conditioning.time_signature)
    structural_control_vocabulary = StructuralControlVocabulary(resolved_model_config.conditioning.structural)
    _LOGGER.info("Building train/validation DataLoaders")
    started_at = perf_counter()
    train_loader, validation_loader = build_dataloaders(
        split,
        batch_size=training_config.optimization.batch_size,
        shuffle_train=True,
        num_workers=training_config.runtime.num_workers,
        conditioning=training_config.conditioning,
        include_structural_controls=training_config.conditioning.use_structural_conditioning,
        time_signature_vocabulary=time_signature_vocabulary,
        token_vocabulary=vocabulary,
        musical_auxiliary_targets=resolved_model_config.musical_auxiliary_targets,
        structural_control_vocabulary=structural_control_vocabulary,
        max_sequence_length=resolved_model_config.transformer.max_sequence_length,
    )
    _LOGGER.info("Built train/validation DataLoaders in %.1fs", perf_counter() - started_at)
    figure_profile_artifacts = (
        load_generation_figure_profile_artifacts(
            source_directory=source_directory,
            ingestion_config=ingestion_config,
            tokenization_config=tokenization_config,
        )
        if training_config.generation_evaluation.enabled
        else None
    )
    _LOGGER.info("Initializing model")
    started_at = perf_counter()
    model = HierarchicalAutoregressiveModel(resolved_model_config)
    _LOGGER.info("Initialized model in %.1fs", perf_counter() - started_at)
    tracker = build_training_tracker(training_config=training_config)
    with tracker:
        tracker.log_setup(
            training_config=training_config,
            model_config=resolved_model_config,
            split=split,
        )
        tracker.log_split_figure_metrics(
            metrics={
                **split_figure_profile_metrics(
                    split,
                    token_vocabulary=vocabulary,
                    tokenization_config=tokenization_config,
                    workers=training_config.runtime.num_workers,
                    show_progress=show_progress,
                ),
                **split_musical_auxiliary_profile_metrics(
                    split,
                    token_vocabulary=vocabulary,
                    target_config=resolved_model_config.musical_auxiliary_targets,
                ),
            }
        )
        trainer = PretrainingTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader,
            validation_loader=validation_loader,
            tracker=tracker,
            show_progress=show_progress,
            token_kind_ids=build_token_kind_ids(vocabulary),
            token_attribute_lookup=build_token_attribute_lookup(vocabulary),
            validity_mask_builder=TrainingValidityMaskBuilder(vocabulary),
            generation_evaluator=GenerationSuiteEvaluator(
                config=training_config.generation_evaluation,
                conditioning=training_config.conditioning,
                model_config=resolved_model_config,
                token_vocabulary=vocabulary,
                duration_vocabulary=vocabulary.duration_vocabulary,
                include_bar_count_control=False,
                figure_profile_artifacts=figure_profile_artifacts,
            ),
        )
        return trainer.train(invalid_files=split.invalid_files)


def _move_batch_to_device(batch: TrainingBatch, *, device: torch.device) -> TrainingBatch:
    difficulty_ids = batch.difficulty_ids.to(device) if batch.difficulty_ids is not None else None
    return TrainingBatch(
        input_token_ids=batch.input_token_ids.to(device),
        target_token_ids=batch.target_token_ids.to(device),
        target_token_attributes=batch.target_token_attributes.to(device),
        musical_auxiliary_targets=batch.musical_auxiliary_targets.to(device),
        bar_positions=batch.bar_positions.to(device),
        target_bar_positions=batch.target_bar_positions.to(device),
        bar_relative_ticks=batch.bar_relative_ticks.to(device),
        bar_duration_ticks=batch.bar_duration_ticks.to(device),
        active_hand_ids=batch.active_hand_ids.to(device),
        structural_control_ids=batch.structural_control_ids.to(device),
        scale_roots=batch.scale_roots.to(device),
        scale_type_ids=batch.scale_type_ids.to(device),
        time_numerators=batch.time_numerators.to(device),
        time_denominators=batch.time_denominators.to(device),
        bar_counts=batch.bar_counts.to(device),
        bar_durations=batch.bar_durations,
        token_padding_mask=batch.token_padding_mask.to(device),
        difficulty_ids=difficulty_ids,
        conditioning_scale_type_ids=batch.conditioning_scale_type_ids.to(device),
        conditioning_time_signature_ids=batch.conditioning_time_signature_ids.to(device),
    )


def _validation_loss_improved(
    *,
    validation_loss: float,
    best_validation_loss: float | None,
    min_delta: float,
) -> bool:
    if best_validation_loss is None:
        return True

    return validation_loss < best_validation_loss - min_delta
