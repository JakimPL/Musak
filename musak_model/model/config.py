from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.common.files import load_yaml_config
from musak_model.conditioning.config import CONDITIONING_CONFIG_PATH, ConditioningConfig
from musak_model.paths import CONFIGS_DIR

MODEL_CONFIG_DIR: Final[Path] = CONFIGS_DIR / "model"


class CNNConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    out_channels: int = Field(ge=1)
    kernel_sizes: tuple[int, ...] = Field(min_length=1)
    num_layers: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)


class GRUConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_size: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    bidirectional: bool


class TransformerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_size: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    feedforward_size: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    max_sequence_length: int = Field(ge=1)


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    vocab_size: int = Field(ge=1)
    cnn: CNNConfig
    gru: GRUConfig
    transformer: TransformerConfig
    conditioning: ConditioningConfig

    @classmethod
    def load(
        cls,
        *,
        vocab_size: int,
        config_dir: Path = MODEL_CONFIG_DIR,
        conditioning_config_path: Path = CONDITIONING_CONFIG_PATH,
    ) -> "ModelConfig":
        return cls(
            vocab_size=vocab_size,
            cnn=CNNConfig.model_validate(load_yaml_config(config_dir / "cnn.yml")),
            gru=GRUConfig.model_validate(load_yaml_config(config_dir / "gru.yml")),
            transformer=TransformerConfig.model_validate(load_yaml_config(config_dir / "transformer.yml")),
            conditioning=ConditioningConfig.load(conditioning_config_path),
        )
