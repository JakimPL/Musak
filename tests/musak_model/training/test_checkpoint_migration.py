from pathlib import Path
from typing import Final

import pytest
import torch
from torch.optim import AdamW

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.conditioning.config import (
    ConditioningConfig,
    DifficultyConfig,
    HarmonicConditioningConfig,
    HarmonicFusionMode,
)
from musak_model.conditioning.time_signature import TimeSignatureVocabularyConfig
from musak_model.model import HierarchicalAutoregressiveModel
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
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint import save_checkpoint
from musak_model.training.checkpoint_migration import CheckpointMigrationError, migrate_checkpoint_to_model

HIDDEN_SIZE: Final[int] = 16
FLAT_INPUT_EMBEDDING_WEIGHT_KEY: Final[str] = "_input_embeddings._flat_embedding.weight"


def _musical_auxiliary_target_config() -> MusicalAuxiliaryTargetConfig:
    return MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
        rhythmic_diversity_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        voice_independence_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        hand_span_bucket_boundaries=(3, 5, 8, 12, 16),
    )


def _token_vocabulary() -> TokenVocabulary:
    tokenization_config = TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)
    return TokenVocabulary(DurationVocabulary(tokenization_config))


def _small_model_config() -> ModelConfig:
    return ModelConfig(
        vocabulary_size=_token_vocabulary().vocabulary_size,
        duration_vocabulary_size=_token_vocabulary().duration_vocabulary.vocabulary_size(),
        input=ModelInputConfig(embedding_mode=TokenInputEmbeddingMode.FLAT),
        output=ModelOutputConfig(mode=ModelOutputMode.FLAT),
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
            harmony=_harmony_config(enabled=False),
            cfg_dropout_probability=0.0,
        ),
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


def test_migrate_checkpoint_initializes_missing_target_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    old_model = HierarchicalAutoregressiveModel(_small_model_config())
    checkpoint_path = tmp_path / "old.pt"
    save_checkpoint(
        checkpoint_path,
        model=old_model,
        optimizer=AdamW(old_model.parameters(), lr=0.001),
        epoch=3,
        best_validation_loss=1.5,
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    missing_key = "_structural_control_embeddings.7.weight"
    checkpoint["model_state_dict"].pop(missing_key)
    checkpoint["model_state_dict"]["unused.weight"] = torch.zeros(1)
    torch.save(checkpoint, checkpoint_path)

    torch.manual_seed(17)
    target_model = HierarchicalAutoregressiveModel(_small_model_config())
    output_path = tmp_path / "adapted.pt"
    report = migrate_checkpoint_to_model(
        checkpoint_path,
        output_path,
        model=target_model,
        device=torch.device("cpu"),
    )

    migrated = torch.load(output_path, map_location=torch.device("cpu"))
    migrated_state = migrated["model_state_dict"]
    assert torch.equal(migrated_state[missing_key], target_model.state_dict()[missing_key])
    assert torch.equal(
        migrated_state[FLAT_INPUT_EMBEDDING_WEIGHT_KEY],
        old_model.state_dict()[FLAT_INPUT_EMBEDDING_WEIGHT_KEY],
    )
    assert migrated["optimizer_state_dict"] == {}
    assert report.ignored_source_keys == ("unused.weight",)
    assert [tensor.key for tensor in report.changed_tensors] == [missing_key]


def test_migrate_checkpoint_rejects_truncation_without_explicit_permission(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = HierarchicalAutoregressiveModel(_small_model_config())
    checkpoint_path = tmp_path / "old.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=AdamW(model.parameters(), lr=0.001),
        epoch=0,
        best_validation_loss=None,
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    key = "_structural_control_embeddings.7.weight"
    checkpoint["model_state_dict"][key] = torch.zeros(
        model.state_dict()[key].size(0) + 1,
        model.state_dict()[key].size(1),
    )
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(CheckpointMigrationError, match="refusing to truncate"):
        migrate_checkpoint_to_model(
            checkpoint_path,
            tmp_path / "adapted.pt",
            model=model,
            device=torch.device("cpu"),
        )
