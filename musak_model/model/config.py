from pydantic import BaseModel, ConfigDict, Field


class CNNConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    out_channels: int = 256
    kernel_sizes: tuple[int, ...] = (3, 5)
    num_layers: int = 3
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class GRUConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    bidirectional: bool = False


class TransformerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    feedforward_size: int = 2048
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_sequence_length: int = 1024


class ConditioningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_difficulty_levels: int = 6
    num_scale_types: int = 9
    num_time_signatures: int = 5
    cfg_dropout_probability: float = Field(default=0.1, ge=0.0, lt=1.0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    vocab_size: int
    cnn: CNNConfig = CNNConfig()
    gru: GRUConfig = GRUConfig()
    transformer: TransformerConfig = TransformerConfig()
    conditioning: ConditioningConfig = ConditioningConfig()
