from pathlib import Path
from typing import Final

import pytest
import torch
from torch.optim import AdamW

from musak_model.conditioning.config import ConditioningConfig, DifficultyConfig
from musak_model.conditioning.time_signature import TimeSignatureVocabularyConfig
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import CNNConfig, GRUConfig, ModelConfig, TransformerConfig
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint import save_checkpoint
from musak_model.training.checkpoint_migration import CheckpointMigrationError, migrate_checkpoint_to_model

HIDDEN_SIZE: Final[int] = 16


def _token_vocabulary() -> TokenVocabulary:
    tokenization_config = TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)
    return TokenVocabulary(DurationVocabulary(tokenization_config))


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
        migrated_state["_token_embedding.weight"],
        old_model.state_dict()["_token_embedding.weight"],
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
