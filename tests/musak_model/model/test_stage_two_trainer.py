from fractions import Fraction
from pathlib import Path
from typing import Final

import torch
from torch.optim import AdamW

from musak_model.conditioning.config import ConditioningConfig, DifficultyConfig
from musak_model.conditioning.time_signature import TimeSignatureVocabularyConfig
from musak_model.data.config import SegmentationConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import CNNConfig, GRUConfig, ModelConfig, TransformerConfig
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint import save_checkpoint
from musak_model.training.config import (
    MlflowConfig,
    OptimizationConfig,
    RuntimeConfig,
    StageTwoCheckpointConfig,
    StageTwoTrainingConfig,
    TrainingConditioningConfig,
)
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit
from musak_model.training.stage_two import train_stage_two

HIDDEN_SIZE: Final[int] = 16


def _tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)


def _token_vocabulary() -> TokenVocabulary:
    return TokenVocabulary(DurationVocabulary(_tokenization_config()))


def _small_model_config() -> ModelConfig:
    return ModelConfig(
        vocabulary_size=_token_vocabulary().vocabulary_size,
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


def _sample() -> EncodedExercise:
    token_vocabulary = _token_vocabulary()
    quarter_id = token_vocabulary.duration_vocabulary.fraction_to_id(Fraction(1, 4))
    token_ids = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        ]
    )
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=[0, 0],
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


def test_train_stage_two_loads_stage_one_checkpoint_and_runs_epoch(tmp_path: Path, monkeypatch) -> None:
    model_config = _small_model_config()
    stage_one_model = HierarchicalAutoregressiveModel(model_config)
    optimizer = AdamW(stage_one_model.parameters(), lr=0.001)
    stage_one_checkpoint = tmp_path / "stage_one.pt"
    save_checkpoint(
        stage_one_checkpoint,
        model=stage_one_model,
        optimizer=optimizer,
        epoch=0,
        best_validation_loss=None,
    )
    split = IngestionSplit(train=[_sample(), _sample()], validation=[_sample()], invalid_files=[])
    monkeypatch.setattr("musak_model.training.stage_two.build_split", lambda *args, **kwargs: split)

    result = train_stage_two(
        tmp_path,
        ingestion_config=IngestionConfig(validation_fraction=0.0, split_seed=1, processed_root=None),
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1),
        training_config=StageTwoTrainingConfig(
            optimization=OptimizationConfig(epochs=1, batch_size=2, learning_rate=0.001, weight_decay=0.0),
            runtime=RuntimeConfig(num_workers=0, device="cpu"),
            checkpoints=StageTwoCheckpointConfig(
                checkpoint_dir=tmp_path / "stage_two",
                stage_one_checkpoint=stage_one_checkpoint,
            ),
            conditioning=TrainingConditioningConfig(
                use_time_signature=True,
                use_scale_type=True,
                use_structural_conditioning=True,
            ),
            mlflow=MlflowConfig(enable_mlflow=False),
        ),
        tokenization_config=_tokenization_config(),
        model_config=model_config,
    )

    assert len(result.metrics) == 1
    assert result.latest_checkpoint_path == tmp_path / "stage_two" / "latest.pt"
    assert result.latest_checkpoint_path.exists()
