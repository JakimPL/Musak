from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.data.config import SegmentationConfig
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import ModelConfig
from musak_model.paths import CONDITIONING_CONFIG_PATH
from musak_model.tokens.vocabulary import build_default_token_vocabulary
from musak_model.training.checkpoint import load_checkpoint, save_checkpoint
from musak_model.training.config import TrainingConfig
from musak_model.training.dataset import TrainingBatch, build_dataloaders
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.schema import IngestionErrorRecord
from musak_model.training.ingestion.split import build_split
from musak_model.training.tracking import NoOpTrainingTracker, TrainingTracker, build_training_tracker


class EpochMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: int
    train_loss: float
    validation_loss: float | None


class TrainingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    metrics: list[EpochMetrics]
    best_checkpoint_path: Path | None
    latest_checkpoint_path: Path | None
    invalid_files: list[IngestionErrorRecord]


class StageOneTrainer:
    def __init__(
        self,
        *,
        model: HierarchicalAutoregressiveModel,
        config: TrainingConfig,
        train_loader: DataLoader[TrainingBatch],
        validation_loader: DataLoader[TrainingBatch],
        tracker: TrainingTracker | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._device = torch.device(config.device)
        self._optimizer = AdamW(
            self._model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._model.to(self._device)
        self._tracker = tracker or NoOpTrainingTracker()

    def train(self, *, invalid_files: list[IngestionErrorRecord] | None = None) -> TrainingResult:
        if len(self._train_loader) == 0:
            raise ValueError("training loader is empty")

        start_epoch = 0
        best_validation_loss: float | None = None
        if self._config.resume_checkpoint is not None:
            start_epoch, best_validation_loss = load_checkpoint(
                self._config.resume_checkpoint,
                model=self._model,
                optimizer=self._optimizer,
                device=self._device,
            )

        metrics: list[EpochMetrics] = []
        best_checkpoint_path: Path | None = None
        latest_checkpoint_path = self._config.checkpoint_dir / "latest.pt"

        for epoch in range(start_epoch, self._config.epochs):
            train_loss = self._train_epoch()
            validation_loss = self._validate_epoch()
            metric = EpochMetrics(epoch=epoch, train_loss=train_loss, validation_loss=validation_loss)
            metrics.append(metric)
            self._tracker.log_epoch(epoch=epoch, train_loss=train_loss, validation_loss=validation_loss)

            save_checkpoint(
                latest_checkpoint_path,
                model=self._model,
                optimizer=self._optimizer,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
            )

            score = validation_loss if validation_loss is not None else train_loss
            if best_validation_loss is None or score < best_validation_loss:
                best_validation_loss = score
                best_checkpoint_path = self._config.checkpoint_dir / "best.pt"
                save_checkpoint(
                    best_checkpoint_path,
                    model=self._model,
                    optimizer=self._optimizer,
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                )

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

    def _train_epoch(self) -> float:
        self._model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in self._train_loader:
            batch = _move_batch_to_device(batch, device=self._device)
            self._optimizer.zero_grad(set_to_none=True)
            loss, token_count = self._loss_for_batch(batch)
            loss.backward()  # type: ignore[no-untyped-call]
            self._optimizer.step()
            total_loss += float(loss.detach().item()) * token_count
            total_tokens += token_count

        return total_loss / total_tokens

    def _validate_epoch(self) -> float | None:
        if len(self._validation_loader) == 0:
            return None

        self._model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in self._validation_loader:
                batch = _move_batch_to_device(batch, device=self._device)
                loss, token_count = self._loss_for_batch(batch)
                total_loss += float(loss.detach().item()) * token_count
                total_tokens += token_count

        return total_loss / total_tokens

    def _loss_for_batch(self, batch: TrainingBatch) -> tuple[Tensor, int]:
        logits = self._model(
            batch.input_token_ids,
            bar_positions=batch.bar_positions,
            difficulty_ids=batch.difficulty_ids if self._config.use_conditioning else None,
            scale_type_ids=batch.scale_type_ids if self._config.use_conditioning else None,
            time_signature_ids=batch.time_signature_ids if self._config.use_conditioning else None,
            token_padding_mask=batch.token_padding_mask,
        )
        flat_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.target_token_ids.reshape(-1),
            reduction="none",
        )
        valid_mask = ~batch.token_padding_mask.reshape(-1)
        token_count = int(valid_mask.sum().item())
        if token_count == 0:
            raise ValueError("batch has no valid target tokens")

        return (flat_loss[valid_mask].sum() / token_count), token_count


def train_stage_one(
    source_dir: Path,
    *,
    ingestion_config: IngestionConfig,
    segmentation_config: SegmentationConfig,
    training_config: TrainingConfig,
    model_config: ModelConfig | None = None,
    conditioning_config_path: Path = CONDITIONING_CONFIG_PATH,
) -> TrainingResult:
    split = build_split(source_dir, config=ingestion_config, segmentation=segmentation_config)
    vocabulary = build_default_token_vocabulary()
    resolved_model_config = model_config or ModelConfig.load(
        vocabulary_size=vocabulary.vocabulary_size,
        conditioning_config_path=conditioning_config_path,
    )
    time_signature_vocabulary = TimeSignatureVocabulary(resolved_model_config.conditioning.time_signature)
    train_loader, validation_loader = build_dataloaders(
        split,
        batch_size=training_config.batch_size,
        shuffle_train=True,
        num_workers=training_config.num_workers,
        include_conditioning=training_config.use_conditioning,
        time_signature_vocabulary=time_signature_vocabulary,
    )
    model = HierarchicalAutoregressiveModel(resolved_model_config)
    tracker = build_training_tracker(training_config=training_config)
    with tracker:
        tracker.log_setup(
            training_config=training_config,
            model_config=resolved_model_config,
            split=split,
        )
        trainer = StageOneTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader,
            validation_loader=validation_loader,
            tracker=tracker,
        )
        return trainer.train(invalid_files=split.invalid_files)


def _move_batch_to_device(batch: TrainingBatch, *, device: torch.device) -> TrainingBatch:
    difficulty_ids = batch.difficulty_ids.to(device) if batch.difficulty_ids is not None else None
    return TrainingBatch(
        input_token_ids=batch.input_token_ids.to(device),
        target_token_ids=batch.target_token_ids.to(device),
        bar_positions=batch.bar_positions.to(device),
        token_padding_mask=batch.token_padding_mask.to(device),
        difficulty_ids=difficulty_ids,
        scale_type_ids=batch.scale_type_ids.to(device),
        time_signature_ids=batch.time_signature_ids.to(device),
    )
