from fractions import Fraction
from pathlib import Path
from typing import Final

import pytest
from torch.optim import AdamW

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.conditioning.config import ConditioningConfig, DifficultyConfig
from musak_model.conditioning.time_signature import TimeSignatureVocabularyConfig
from musak_model.data.config import SegmentationConfig
from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import (
    CNNConfig,
    GRUConfig,
    ModelConfig,
    ModelOutputConfig,
    ModelOutputMode,
    TransformerConfig,
)
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint import save_checkpoint
from musak_model.training.config import (
    EventObjectiveConfig,
    FinetuningCheckpointConfig,
    FinetuningTrainingConfig,
    GenerationEvaluationConfig,
    MlflowConfig,
    MusicalAuxiliaryObjectiveConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainingConditioningConfig,
)
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit
from musak_model.training.stages.finetuning import finetune

HIDDEN_SIZE: Final[int] = 16


def _musical_auxiliary_target_config() -> MusicalAuxiliaryTargetConfig:
    return MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
        rhythmic_diversity_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        voice_independence_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        hand_span_bucket_boundaries=(3, 5, 8, 12, 16),
    )


def _scale_matcher_config() -> ScaleMatcherConfig:
    return ScaleMatcherConfig(
        support_score_margin=0.08,
        selection_score_margin=0.03,
        maximum_unexplained_weight_fraction=0.10,
        maximum_explanation_pitch_class_count=9,
    )


def _tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)


def _token_vocabulary() -> TokenVocabulary:
    return TokenVocabulary(DurationVocabulary(_tokenization_config()))


def _small_model_config() -> ModelConfig:
    token_vocabulary = _token_vocabulary()
    return ModelConfig(
        vocabulary_size=token_vocabulary.vocabulary_size,
        duration_vocabulary_size=token_vocabulary.duration_vocabulary.vocabulary_size(),
        output=ModelOutputConfig(mode=ModelOutputMode.FACTORIZED),
        musical_auxiliary_targets=_musical_auxiliary_target_config(),
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
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
            difficulty_level=1,
        ),
    )


def test_train_finetuning_loads_pretraining_checkpoint_and_runs_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = _small_model_config()
    pretraining_model = HierarchicalAutoregressiveModel(model_config)
    optimizer = AdamW(pretraining_model.parameters(), lr=0.001)
    pretraining_checkpoint = tmp_path / "pretraining.pt"
    save_checkpoint(
        pretraining_checkpoint,
        model=pretraining_model,
        optimizer=optimizer,
        epoch=0,
        best_validation_loss=None,
    )
    split = IngestionSplit(train=[_sample(), _sample()], validation=[_sample()], invalid_files=[])
    monkeypatch.setattr("musak_model.training.stages.finetuning.build_split", lambda *args, **kwargs: split)

    result = finetune(
        tmp_path,
        ingestion_config=IngestionConfig(
            validation_fraction=0.0,
            split_seed=1,
            scale_matcher=_scale_matcher_config(),
            processed_root=None,
        ),
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1),
        training_config=FinetuningTrainingConfig(
            optimization=OptimizationConfig(epochs=1, batch_size=2, learning_rate=0.001, weight_decay=0.0),
            runtime=RuntimeConfig(num_workers=1, device="cpu"),
            checkpoints=FinetuningCheckpointConfig(
                checkpoint_directory=tmp_path / "finetuning",
                pretraining_checkpoint=pretraining_checkpoint,
            ),
            event_objective=EventObjectiveConfig(
                mode=ModelOutputMode.FACTORIZED,
                kind_weight=1.0,
                duration_weight=1.0,
                degree_weight=1.0,
                accidental_weight=1.0,
                octave_offset_weight=1.0,
                hand_weight=1.0,
            ),
            musical_auxiliary_targets=_musical_auxiliary_target_config(),
            musical_auxiliary_objective=MusicalAuxiliaryObjectiveConfig(
                enabled=True,
                weight=0.1,
                bar_weight=1.0,
                note_density_weight=1.0,
                rhythmic_diversity_weight=1.0,
                voice_independence_weight=1.0,
                uses_accidentals_weight=1.0,
                dotted_duration_weight=1.0,
                hand_span_weight=1.0,
            ),
            conditioning=TrainingConditioningConfig(
                use_time_signature=True,
                use_scale_type=True,
                use_difficulty=False,
                use_structural_conditioning=True,
                use_validity_penalty=False,
                validity_penalty_weight=0.05,
            ),
            mlflow=MlflowConfig(enable_mlflow=False),
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
        ),
        tokenization_config=_tokenization_config(),
        model_config=model_config,
    )

    assert len(result.metrics) == 1
    assert result.latest_checkpoint_path == tmp_path / "finetuning" / "latest.pt"
    assert result.latest_checkpoint_path.exists()
    assert (tmp_path / "finetuning" / "epoch_0000.pt").exists()
