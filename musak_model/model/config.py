from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.conditioning.config import ConditioningConfig
from musak_model.paths import CONDITIONING_CONFIG_PATH, MODEL_CONFIG_DIRECTORY
from musak_shared.files import load_yaml_config


class CNNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    out_channels: int = Field(ge=1)
    kernel_sizes: tuple[int, ...] = Field(min_length=1)
    num_layers: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)


class GRUConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    hidden_size: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    bidirectional: bool


class TransformerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_size: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    feedforward_size: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    max_sequence_length: int = Field(ge=1)


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    vocabulary_size: int = Field(ge=1)
    cnn: CNNConfig
    gru: GRUConfig
    transformer: TransformerConfig
    conditioning: ConditioningConfig

    @classmethod
    def load(
        cls,
        *,
        vocabulary_size: int,
        config_directory: Path = MODEL_CONFIG_DIRECTORY,
        conditioning_config_path: Path = CONDITIONING_CONFIG_PATH,
    ) -> ModelConfig:
        return cls(
            vocabulary_size=vocabulary_size,
            cnn=CNNConfig.model_validate(load_yaml_config(config_directory / "cnn.yml")),
            gru=GRUConfig.model_validate(load_yaml_config(config_directory / "gru.yml")),
            transformer=TransformerConfig.model_validate(load_yaml_config(config_directory / "transformer.yml")),
            conditioning=ConditioningConfig.load(conditioning_config_path),
        )
