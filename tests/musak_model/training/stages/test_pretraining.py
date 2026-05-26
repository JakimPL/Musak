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
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.config import (
    CheckpointConfig,
    GenerationEvaluationConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainingConditioningConfig,
    TrainingConfig,
)
from musak_model.training.dataset.collate import collate_training_examples
from musak_model.training.dataset.examples import EncodedExerciseDataset
from musak_model.training.dataset.schema import TrainingBatch
from musak_model.training.ingestion.schema import EncodedExercise, IngestionErrorRecord, IngestionSplit
from musak_model.training.metrics import EpochMetrics
from musak_model.training.stages.pretraining import PretrainingTrainer
from musak_model.training.validity import TrainingValidityMaskBuilder

HIDDEN_SIZE: Final[int] = 16


class FakeTracker:
    def __init__(self) -> None:
        self.epochs: list[int] = []
        self.generation_evaluations: list[tuple[int, dict[str, float]]] = []
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

    def log_epoch(self, *, metrics: EpochMetrics) -> None:
        self.epochs.append(metrics.epoch)

    def log_generation_evaluation(self, *, metrics: dict[str, float], epoch: int) -> None:
        self.generation_evaluations.append((epoch, metrics))

    def log_checkpoints(self, *, latest_checkpoint_path: Path | None, best_checkpoint_path: Path | None) -> None:
        self.checkpoint_logged = True

    def log_invalid_files(self, *, invalid_files: list[IngestionErrorRecord]) -> None:
        self.invalid_files_logged = True


def _small_model_config() -> ModelConfig:
    return ModelConfig(
        vocabulary_size=_token_vocabulary().vocabulary_size,
        cnn=CNNConfig(enabled=True, out_channels=HIDDEN_SIZE, kernel_sizes=(3,), num_layers=1, dropout=0.0),
        gru=GRUConfig(enabled=True, hidden_size=HIDDEN_SIZE, num_layers=1, dropout=0.0, bidirectional=False),
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


def _training_config(
    checkpoint_directory: Path,
    *,
    resume_checkpoint: Path | None = None,
    epochs: int = 1,
    conditioning: TrainingConditioningConfig | None = None,
    save_all_epochs: bool = False,
) -> TrainingConfig:
    return TrainingConfig(
        optimization=OptimizationConfig(epochs=epochs, batch_size=2, learning_rate=0.001, weight_decay=0.0),
        runtime=RuntimeConfig(num_workers=1, device="cpu"),
        conditioning=conditioning if conditioning is not None else _conditioning_config(),
        checkpoints=CheckpointConfig(
            checkpoint_directory=checkpoint_directory,
            resume_checkpoint=resume_checkpoint,
            save_all_epochs=save_all_epochs,
        ),
        generation_evaluation=_generation_evaluation_config(enabled=False),
    )


def _conditioning_config(
    *,
    use_validity_penalty: bool = False,
    validity_penalty_weight: float = 0.05,
) -> TrainingConditioningConfig:
    return TrainingConditioningConfig(
        use_time_signature=False,
        use_scale_type=False,
        use_difficulty=False,
        use_structural_conditioning=False,
        use_validity_penalty=use_validity_penalty,
        validity_penalty_weight=validity_penalty_weight,
    )


def _generation_evaluation_config(*, enabled: bool) -> GenerationEvaluationConfig:
    return GenerationEvaluationConfig(
        enabled=enabled,
        every_epochs=5,
        soft_sample_count=1,
        hard_sample_count=0,
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
    )


class FakeGenerationEvaluator:
    def __init__(self) -> None:
        self.epochs_seen = 0

    def evaluate(self, model: HierarchicalAutoregressiveModel, *, device: torch.device) -> dict[str, float]:
        self.epochs_seen += 1
        return {"generation/soft/count/samples": 1.0}


def _sample(token_ids: list[int], bar_positions: list[int]) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=bar_positions,
        metadata=SegmentMetadata(
            scale_root=0,
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
    token_vocabulary = _token_vocabulary()
    dataset = EncodedExerciseDataset(
        [
            _sample([1, 2, 3, 4], [0, 0, 0, 0]),
            _sample([2, 3, 4, 5], [0, 0, 0, 0]),
        ],
        time_signature_vocabulary=TimeSignatureVocabulary(
            TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2)
        ),
        token_vocabulary=token_vocabulary,
        conditioning=_conditioning_config(),
    )
    return DataLoader(dataset, batch_size=2, collate_fn=collate_training_examples)


def _token_vocabulary() -> TokenVocabulary:
    tokenization_config = TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)
    return TokenVocabulary(DurationVocabulary(tokenization_config))


def test_trainer_runs_one_epoch_and_writes_checkpoints(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = HierarchicalAutoregressiveModel(_small_model_config())
    trainer = PretrainingTrainer(
        model=model,
        config=_training_config(tmp_path),
        train_loader=_loader(),
        validation_loader=_loader(),
    )

    result = trainer.train()

    assert len(result.metrics) == 1
    assert result.metrics[0].train_loss > 0
    assert result.metrics[0].train_perplexity > 1
    assert 0 <= result.metrics[0].train_token_accuracy <= 1
    assert result.metrics[0].validation_loss is not None
    assert result.metrics[0].validation_perplexity is not None
    assert result.metrics[0].validation_token_accuracy is not None
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert not (tmp_path / "epoch_0000.pt").exists()


def test_trainer_logs_to_tracker(tmp_path: Path) -> None:
    tracker = FakeTracker()
    model = HierarchicalAutoregressiveModel(_small_model_config())
    trainer = PretrainingTrainer(
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


def test_trainer_saves_epoch_checkpoints_when_enabled(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = HierarchicalAutoregressiveModel(_small_model_config())
    trainer = PretrainingTrainer(
        model=model,
        config=_training_config(tmp_path, epochs=2, save_all_epochs=True),
        train_loader=_loader(),
        validation_loader=_loader(),
    )

    trainer.train()

    assert (tmp_path / "epoch_0000.pt").exists()
    assert (tmp_path / "epoch_0001.pt").exists()


def test_trainer_logs_generation_evaluation_on_configured_cadence(tmp_path: Path) -> None:
    tracker = FakeTracker()
    evaluator = FakeGenerationEvaluator()
    model = HierarchicalAutoregressiveModel(_small_model_config())
    config = _training_config(tmp_path, epochs=5).model_copy(
        update={"generation_evaluation": _generation_evaluation_config(enabled=True)}
    )
    trainer = PretrainingTrainer(
        model=model,
        config=config,
        train_loader=_loader(),
        validation_loader=_loader(),
        tracker=tracker,
        generation_evaluator=evaluator,
    )

    trainer.train()

    assert evaluator.epochs_seen == 1
    assert tracker.generation_evaluations == [(4, {"generation/soft/count/samples": 1.0})]


def test_trainer_resumes_from_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    first_model = HierarchicalAutoregressiveModel(_small_model_config())
    first_trainer = PretrainingTrainer(
        model=first_model,
        config=_training_config(tmp_path),
        train_loader=_loader(),
        validation_loader=_loader(),
    )
    first_trainer.train()

    second_model = HierarchicalAutoregressiveModel(_small_model_config())
    second_trainer = PretrainingTrainer(
        model=second_model,
        config=_training_config(tmp_path, resume_checkpoint=tmp_path / "latest.pt", epochs=2),
        train_loader=_loader(),
        validation_loader=_loader(),
    )

    result = second_trainer.train()

    assert [metric.epoch for metric in result.metrics] == [1]


def test_trainer_resumes_from_model_only_checkpoint_with_fresh_optimizer(tmp_path: Path) -> None:
    torch.manual_seed(0)
    first_model = HierarchicalAutoregressiveModel(_small_model_config())
    first_trainer = PretrainingTrainer(
        model=first_model,
        config=_training_config(tmp_path),
        train_loader=_loader(),
        validation_loader=_loader(),
    )
    first_trainer.train()
    latest_checkpoint = tmp_path / "latest.pt"
    checkpoint = torch.load(latest_checkpoint, map_location=torch.device("cpu"))
    checkpoint["optimizer_state_dict"] = {}
    torch.save(checkpoint, latest_checkpoint)

    second_model = HierarchicalAutoregressiveModel(_small_model_config())
    second_trainer = PretrainingTrainer(
        model=second_model,
        config=_training_config(tmp_path, resume_checkpoint=latest_checkpoint, epochs=2),
        train_loader=_loader(),
        validation_loader=_loader(),
    )

    result = second_trainer.train()

    assert [metric.epoch for metric in result.metrics] == [1]


def test_trainer_resumes_from_checkpoint_with_incompatible_optimizer_state(tmp_path: Path) -> None:
    torch.manual_seed(0)
    first_model = HierarchicalAutoregressiveModel(_small_model_config())
    first_trainer = PretrainingTrainer(
        model=first_model,
        config=_training_config(tmp_path),
        train_loader=_loader(),
        validation_loader=_loader(),
    )
    first_trainer.train()
    latest_checkpoint = tmp_path / "latest.pt"
    checkpoint = torch.load(latest_checkpoint, map_location=torch.device("cpu"))
    checkpoint["optimizer_state_dict"]["param_groups"][0]["params"].append(-1)
    torch.save(checkpoint, latest_checkpoint)

    second_model = HierarchicalAutoregressiveModel(_small_model_config())
    second_trainer = PretrainingTrainer(
        model=second_model,
        config=_training_config(tmp_path, resume_checkpoint=latest_checkpoint, epochs=2),
        train_loader=_loader(),
        validation_loader=_loader(),
    )

    result = second_trainer.train()

    assert [metric.epoch for metric in result.metrics] == [1]


def test_trainer_reports_validity_penalty_metrics(tmp_path: Path) -> None:
    token_vocabulary = _token_vocabulary()
    model = HierarchicalAutoregressiveModel(_small_model_config())
    trainer = PretrainingTrainer(
        model=model,
        config=_training_config(
            tmp_path,
            conditioning=_conditioning_config(use_validity_penalty=True, validity_penalty_weight=0.05),
        ),
        train_loader=_loader(),
        validation_loader=_loader(),
        validity_mask_builder=TrainingValidityMaskBuilder(token_vocabulary),
    )

    result = trainer.train()

    assert result.metrics[0].train_validity_penalty_loss is not None
    assert result.metrics[0].train_invalid_probability_mass is not None
    assert result.metrics[0].train_invalid_target_rate is not None
