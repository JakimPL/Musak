from pathlib import Path
from types import TracebackType
from typing import Final, Self

import torch
from torch.utils.data import DataLoader

from musak_model.conditioning.config import ConditioningConfig, DifficultyConfig
from musak_model.conditioning.time_signature import TimeSignatureVocabulary, TimeSignatureVocabularyConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import CNNConfig, GRUConfig, ModelConfig, TransformerConfig
from musak_model.tokens.schema import ScaleType
from musak_model.training.config import TrainingConfig
from musak_model.training.dataset import EncodedExerciseDataset, TrainingBatch, collate_training_examples
from musak_model.training.ingestion.schema import EncodedExercise, IngestionErrorRecord, IngestionSplit
from musak_model.training.trainer import StageOneTrainer

VOCAB: Final[int] = 32
HIDDEN_SIZE: Final[int] = 16


class FakeTracker:
    def __init__(self) -> None:
        self.epochs: list[int] = []
        self.checkpoint_logged = False
        self.invalid_files_logged = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def log_setup(self, *, training_config: TrainingConfig, model_config: ModelConfig, split: IngestionSplit) -> None:
        return None

    def log_epoch(self, *, epoch: int, train_loss: float, validation_loss: float | None) -> None:
        self.epochs.append(epoch)

    def log_checkpoints(self, *, latest_checkpoint_path: Path | None, best_checkpoint_path: Path | None) -> None:
        self.checkpoint_logged = True

    def log_invalid_files(self, *, invalid_files: list[IngestionErrorRecord]) -> None:
        self.invalid_files_logged = True


def _small_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB,
        cnn=CNNConfig(out_channels=HIDDEN_SIZE, kernel_sizes=(3,), num_layers=1, dropout=0.0),
        gru=GRUConfig(hidden_size=HIDDEN_SIZE, num_layers=1, dropout=0.0, bidirectional=False),
        transformer=TransformerConfig(
            hidden_size=HIDDEN_SIZE,
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


def _training_config(checkpoint_dir: Path, *, resume_checkpoint: Path | None = None, epochs: int = 1) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        batch_size=2,
        learning_rate=0.001,
        weight_decay=0.0,
        num_workers=0,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=resume_checkpoint,
        device="cpu",
    )


def _sample(token_ids: list[int], bar_positions: list[int]) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=bar_positions,
        metadata=SegmentMetadata(
            key_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
            difficulty_level=1,
        ),
    )


def _loader() -> DataLoader[TrainingBatch]:
    dataset = EncodedExerciseDataset(
        [
            _sample([1, 2, 3, 4], [0, 0, 0, 0]),
            _sample([2, 3, 4, 5], [0, 0, 0, 0]),
        ],
        time_signature_vocabulary=TimeSignatureVocabulary(
            TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2)
        ),
    )
    return DataLoader(dataset, batch_size=2, collate_fn=collate_training_examples)


def test_trainer_runs_one_epoch_and_writes_checkpoints(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = HierarchicalAutoregressiveModel(_small_model_config())
    trainer = StageOneTrainer(
        model=model,
        config=_training_config(tmp_path),
        train_loader=_loader(),
        validation_loader=_loader(),
    )

    result = trainer.train()

    assert len(result.metrics) == 1
    assert result.metrics[0].train_loss > 0
    assert result.metrics[0].validation_loss is not None
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists()


def test_trainer_logs_to_tracker(tmp_path: Path) -> None:
    tracker = FakeTracker()
    model = HierarchicalAutoregressiveModel(_small_model_config())
    trainer = StageOneTrainer(
        model=model,
        config=_training_config(tmp_path),
        train_loader=_loader(),
        validation_loader=_loader(),
        tracker=tracker,
    )

    trainer.train()

    assert tracker.epochs == [0]
    assert tracker.checkpoint_logged is True
    assert tracker.invalid_files_logged is True


def test_trainer_resumes_from_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    first_model = HierarchicalAutoregressiveModel(_small_model_config())
    first_trainer = StageOneTrainer(
        model=first_model,
        config=_training_config(tmp_path),
        train_loader=_loader(),
        validation_loader=_loader(),
    )
    first_trainer.train()

    second_model = HierarchicalAutoregressiveModel(_small_model_config())
    second_trainer = StageOneTrainer(
        model=second_model,
        config=_training_config(tmp_path, resume_checkpoint=tmp_path / "latest.pt", epochs=2),
        train_loader=_loader(),
        validation_loader=_loader(),
    )

    result = second_trainer.train()

    assert [metric.epoch for metric in result.metrics] == [1]
