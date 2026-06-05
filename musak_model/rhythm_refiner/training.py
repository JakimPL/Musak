from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from musak_model.data.config import SegmentationConfig
from musak_model.mlflow import MlflowRun, MlflowRunConfig, flatten_params, write_mlflow_run_id
from musak_model.processing.fingerprint import encoded_samples_fingerprint
from musak_model.rhythm_refiner.config import RhythmRefinerTrainingConfig
from musak_model.rhythm_refiner.dataset import (
    RhythmRefinerBatch,
    RhythmRefinerDataset,
    collate_rhythm_refiner_examples,
    rhythm_refiner_frames_from_samples,
)
from musak_model.rhythm_refiner.model import RhythmRefinerLogits, RhythmRefinerModel
from musak_model.rhythm_refiner.schema import RhythmGridFrame
from musak_model.rhythm_refiner.vocabulary import RHYTHM_TARGET_STATE_COUNT
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint import load_checkpoint, save_checkpoint
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.split import build_split
from musak_model.training.progress import log_split_summary, progress

_LOGGER = logging.getLogger(__name__)
_LATEST_CHECKPOINT_NAME: Final[str] = "latest.pt"
_BEST_CHECKPOINT_NAME: Final[str] = "best.pt"


@dataclass(frozen=True)
class RhythmRefinerEpochMetrics:
    epoch: int
    train_loss: float
    train_activity_loss: float
    train_coactivity_loss: float
    train_activity_accuracy: float
    train_coactivity_accuracy: float
    validation_loss: float
    validation_activity_loss: float
    validation_coactivity_loss: float
    validation_activity_accuracy: float
    validation_coactivity_accuracy: float
    masked_activity_targets: int
    masked_coactivity_targets: int


@dataclass(frozen=True)
class RhythmRefinerTrainingResult:
    metrics: tuple[RhythmRefinerEpochMetrics, ...]
    latest_checkpoint_path: Path
    best_checkpoint_path: Path


@dataclass(frozen=True)
class _MetricAccumulator:
    loss_sum: float = 0.0
    activity_loss_sum: float = 0.0
    coactivity_loss_sum: float = 0.0
    activity_correct: int = 0
    activity_total: int = 0
    coactivity_correct: int = 0
    coactivity_total: int = 0

    def add(
        self,
        *,
        loss: Tensor,
        activity_loss: Tensor,
        coactivity_loss: Tensor,
        activity_correct: int,
        activity_total: int,
        coactivity_correct: int,
        coactivity_total: int,
    ) -> _MetricAccumulator:
        return _MetricAccumulator(
            loss_sum=self.loss_sum + float(loss.detach().cpu()) * max(activity_total, 1),
            activity_loss_sum=self.activity_loss_sum + float(activity_loss.detach().cpu()) * max(activity_total, 1),
            coactivity_loss_sum=self.coactivity_loss_sum
            + float(coactivity_loss.detach().cpu()) * max(coactivity_total, 1),
            activity_correct=self.activity_correct + activity_correct,
            activity_total=self.activity_total + activity_total,
            coactivity_correct=self.coactivity_correct + coactivity_correct,
            coactivity_total=self.coactivity_total + coactivity_total,
        )

    def mean_loss(self) -> float:
        return self.loss_sum / max(self.activity_total, 1)

    def mean_activity_loss(self) -> float:
        return self.activity_loss_sum / max(self.activity_total, 1)

    def mean_coactivity_loss(self) -> float:
        return self.coactivity_loss_sum / max(self.coactivity_total, 1)

    def activity_accuracy(self) -> float:
        return self.activity_correct / max(self.activity_total, 1)

    def coactivity_accuracy(self) -> float:
        return self.coactivity_correct / max(self.coactivity_total, 1)


def train_rhythm_refiner(
    source_directory: Path,
    *,
    ingestion_config: IngestionConfig,
    segmentation_config: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    training_config: RhythmRefinerTrainingConfig,
    show_progress: bool,
) -> RhythmRefinerTrainingResult:
    started_at = perf_counter()
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    split = build_split(
        source_directory,
        config=ingestion_config,
        segmentation=segmentation_config,
        tokenization_config=tokenization_config,
        allow_raw_fallback=True,
        show_progress=show_progress,
    )
    log_split_summary(split)
    train_frames = rhythm_refiner_frames_from_samples(
        split.train,
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        grid_config=training_config.grid,
        data_config=training_config.data,
        show_progress=show_progress,
    )
    validation_frames = rhythm_refiner_frames_from_samples(
        split.validation,
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        grid_config=training_config.grid,
        data_config=training_config.data,
        show_progress=show_progress,
    )
    _LOGGER.info(
        "Built rhythm refiner frame split in %.1fs: train=%s validation=%s",
        perf_counter() - started_at,
        len(train_frames),
        len(validation_frames),
    )
    device = torch.device(training_config.runtime.device)
    train_loader, validation_loader = _dataloaders(
        train_frames=train_frames,
        validation_frames=validation_frames,
        training_config=training_config,
    )
    model = RhythmRefinerModel(training_config.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.optimization.learning_rate,
        weight_decay=training_config.optimization.weight_decay,
    )
    checkpoint_directory = training_config.checkpoints.checkpoint_directory
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = checkpoint_directory / _LATEST_CHECKPOINT_NAME
    best_checkpoint_path = checkpoint_directory / _BEST_CHECKPOINT_NAME
    start_epoch = 1
    best_validation_loss: float | None = None
    if training_config.checkpoints.resume_checkpoint is not None:
        start_epoch, best_validation_loss = load_checkpoint(
            training_config.checkpoints.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    run_name = training_config.mlflow.mlflow_run_name or _default_run_name(
        training_config=training_config,
        train_count=len(train_frames),
        validation_count=len(validation_frames),
    )
    with MlflowRun(
        MlflowRunConfig(
            enabled=training_config.mlflow.enable_mlflow,
            experiment_name=training_config.mlflow.mlflow_experiment_name,
            run_name=run_name,
            run_id=training_config.mlflow.mlflow_run_id,
            tracking_uri=training_config.mlflow.mlflow_tracking_uri,
        )
    ) as run:
        if run.run_id is not None:
            write_mlflow_run_id(checkpoint_directory=checkpoint_directory, run_id=run.run_id)
        _log_setup(
            run,
            training_config=training_config,
            train_count=len(train_frames),
            validation_count=len(validation_frames),
            fingerprint=encoded_samples_fingerprint([*split.train, *split.validation]),
        )
        metrics: list[RhythmRefinerEpochMetrics] = []
        for epoch in range(start_epoch, training_config.optimization.epochs + 1):
            epoch_metrics = _train_epoch(
                model,
                optimizer=optimizer,
                train_loader=train_loader,
                validation_loader=validation_loader,
                training_config=training_config,
                device=device,
                epoch=epoch,
                show_progress=show_progress,
            )
            metrics.append(epoch_metrics)
            run.log_metrics(_mlflow_metric_values(epoch_metrics), step=epoch)
            is_best_epoch = best_validation_loss is None or epoch_metrics.validation_loss < best_validation_loss
            if is_best_epoch:
                best_validation_loss = epoch_metrics.validation_loss
            save_checkpoint(
                latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
            )
            if is_best_epoch:
                save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                )
            if training_config.checkpoints.save_all_epochs:
                save_checkpoint(
                    checkpoint_directory / f"epoch_{epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                )

        run.log_artifact(latest_checkpoint_path, artifact_path="checkpoints")
        run.log_artifact(best_checkpoint_path, artifact_path="checkpoints")

    return RhythmRefinerTrainingResult(
        metrics=tuple(metrics),
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
    )


def _dataloaders(
    *,
    train_frames: tuple[RhythmGridFrame, ...],
    validation_frames: tuple[RhythmGridFrame, ...],
    training_config: RhythmRefinerTrainingConfig,
) -> tuple[DataLoader[RhythmRefinerBatch], DataLoader[RhythmRefinerBatch]]:
    train_dataset = RhythmRefinerDataset(
        train_frames,
        masking=training_config.masking,
        model_config=training_config.model,
    )
    validation_dataset = RhythmRefinerDataset(
        validation_frames,
        masking=training_config.masking,
        model_config=training_config.model,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.optimization.batch_size,
        shuffle=True,
        num_workers=training_config.runtime.num_workers,
        collate_fn=collate_rhythm_refiner_examples,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.optimization.batch_size,
        shuffle=False,
        num_workers=training_config.runtime.num_workers,
        collate_fn=collate_rhythm_refiner_examples,
    )
    return cast(
        tuple[DataLoader[RhythmRefinerBatch], DataLoader[RhythmRefinerBatch]], (train_loader, validation_loader)
    )


def _train_epoch(
    model: RhythmRefinerModel,
    *,
    optimizer: AdamW,
    train_loader: DataLoader[RhythmRefinerBatch],
    validation_loader: DataLoader[RhythmRefinerBatch],
    training_config: RhythmRefinerTrainingConfig,
    device: torch.device,
    epoch: int,
    show_progress: bool,
) -> RhythmRefinerEpochMetrics:
    model.train()
    train_metrics = _MetricAccumulator()
    for batch in progress(
        train_loader,
        description=f"Refiner train epoch {epoch}",
        unit="batch",
        enabled=show_progress,
        total=len(train_loader),
    ):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss, activity_loss, coactivity_loss = _batch_loss(logits, batch, training_config=training_config)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        train_metrics = _accumulate_metrics(
            train_metrics,
            loss=loss,
            activity_loss=activity_loss,
            coactivity_loss=coactivity_loss,
            logits=logits,
            batch=batch,
        )

    model.eval()
    validation_metrics = _MetricAccumulator()
    with torch.no_grad():
        for batch in progress(
            validation_loader,
            description=f"Refiner validation epoch {epoch}",
            unit="batch",
            enabled=show_progress,
            total=len(validation_loader),
        ):
            batch = batch.to(device)
            logits = model(batch)
            loss, activity_loss, coactivity_loss = _batch_loss(logits, batch, training_config=training_config)
            validation_metrics = _accumulate_metrics(
                validation_metrics,
                loss=loss,
                activity_loss=activity_loss,
                coactivity_loss=coactivity_loss,
                logits=logits,
                batch=batch,
            )

    _LOGGER.info(
        "Refiner epoch %s: train_loss=%.6f validation_loss=%.6f train_activity_acc=%.4f validation_activity_acc=%.4f",
        epoch,
        train_metrics.mean_loss(),
        validation_metrics.mean_loss(),
        train_metrics.activity_accuracy(),
        validation_metrics.activity_accuracy(),
    )
    return RhythmRefinerEpochMetrics(
        epoch=epoch,
        train_loss=train_metrics.mean_loss(),
        train_activity_loss=train_metrics.mean_activity_loss(),
        train_coactivity_loss=train_metrics.mean_coactivity_loss(),
        train_activity_accuracy=train_metrics.activity_accuracy(),
        train_coactivity_accuracy=train_metrics.coactivity_accuracy(),
        validation_loss=validation_metrics.mean_loss(),
        validation_activity_loss=validation_metrics.mean_activity_loss(),
        validation_coactivity_loss=validation_metrics.mean_coactivity_loss(),
        validation_activity_accuracy=validation_metrics.activity_accuracy(),
        validation_coactivity_accuracy=validation_metrics.coactivity_accuracy(),
        masked_activity_targets=train_metrics.activity_total,
        masked_coactivity_targets=train_metrics.coactivity_total,
    )


def _batch_loss(
    logits: RhythmRefinerLogits,
    batch: RhythmRefinerBatch,
    *,
    training_config: RhythmRefinerTrainingConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    activity_loss = _masked_cross_entropy(
        logits.activity,
        batch.target_state_ids,
        batch.activity_loss_mask,
        class_count=RHYTHM_TARGET_STATE_COUNT,
    )
    coactivity_loss = _masked_cross_entropy(
        logits.coactivity,
        batch.coactivity_target_ids,
        batch.coactivity_loss_mask,
        class_count=logits.coactivity.size(-1),
    )
    loss = (
        training_config.loss.activity_weight * activity_loss + training_config.loss.coactivity_weight * coactivity_loss
    )
    return loss, activity_loss, coactivity_loss


def _masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    class_count: int,
) -> Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0

    return F.cross_entropy(logits[mask].view(-1, class_count), targets[mask].view(-1))


def _accumulate_metrics(
    accumulator: _MetricAccumulator,
    *,
    loss: Tensor,
    activity_loss: Tensor,
    coactivity_loss: Tensor,
    logits: RhythmRefinerLogits,
    batch: RhythmRefinerBatch,
) -> _MetricAccumulator:
    activity_predictions = logits.activity.argmax(dim=-1)
    activity_mask = batch.activity_loss_mask
    coactivity_predictions = logits.coactivity.argmax(dim=-1)
    coactivity_mask = batch.coactivity_loss_mask
    return accumulator.add(
        loss=loss,
        activity_loss=activity_loss,
        coactivity_loss=coactivity_loss,
        activity_correct=int((activity_predictions[activity_mask] == batch.target_state_ids[activity_mask]).sum()),
        activity_total=int(activity_mask.sum()),
        coactivity_correct=int(
            (coactivity_predictions[coactivity_mask] == batch.coactivity_target_ids[coactivity_mask]).sum()
        ),
        coactivity_total=int(coactivity_mask.sum()),
    )


def _log_setup(
    run: MlflowRun,
    *,
    training_config: RhythmRefinerTrainingConfig,
    train_count: int,
    validation_count: int,
    fingerprint: str,
) -> None:
    if not run.enabled or training_config.mlflow.mlflow_run_id is not None:
        return

    run.log_params(
        flatten_params(
            {
                "refiner": training_config.model_dump(
                    mode="json",
                    exclude={"checkpoints": {"resume_checkpoint"}, "mlflow": {"mlflow_run_id"}},
                ),
                "data": {
                    "train_frames": train_count,
                    "validation_frames": validation_count,
                    "encoded_samples_fingerprint": fingerprint,
                },
            }
        )
    )


def _mlflow_metric_values(metrics: RhythmRefinerEpochMetrics) -> dict[str, float]:
    return {
        "refiner/train/mean/loss": metrics.train_loss,
        "refiner/train/mean/activity_loss": metrics.train_activity_loss,
        "refiner/train/mean/coactivity_loss": metrics.train_coactivity_loss,
        "refiner/train/rate/activity_accuracy": metrics.train_activity_accuracy,
        "refiner/train/rate/coactivity_accuracy": metrics.train_coactivity_accuracy,
        "refiner/train/count/masked_activity_targets": float(metrics.masked_activity_targets),
        "refiner/train/count/masked_coactivity_targets": float(metrics.masked_coactivity_targets),
        "refiner/validation/mean/loss": metrics.validation_loss,
        "refiner/validation/mean/activity_loss": metrics.validation_activity_loss,
        "refiner/validation/mean/coactivity_loss": metrics.validation_coactivity_loss,
        "refiner/validation/rate/activity_accuracy": metrics.validation_activity_accuracy,
        "refiner/validation/rate/coactivity_accuracy": metrics.validation_coactivity_accuracy,
    }


def _default_run_name(
    *,
    training_config: RhythmRefinerTrainingConfig,
    train_count: int,
    validation_count: int,
) -> str:
    return "-".join(
        (
            f"refiner-grid{training_config.grid.grid_denominator}",
            f"e{training_config.optimization.epochs}",
            f"bs{training_config.optimization.batch_size}",
            f"h{training_config.model.hidden_size}",
            f"l{training_config.model.transformer_layers}",
            f"mask{training_config.masking.mask_probability:g}".replace(".", "p"),
            f"tr{train_count}",
            f"va{validation_count}",
        )
    )
