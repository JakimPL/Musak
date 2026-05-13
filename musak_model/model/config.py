from pydantic import BaseModel, ConfigDict, Field


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


class ConditioningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_difficulty_levels: int = Field(ge=1)
    num_scale_types: int = Field(ge=1)
    num_time_signatures: int = Field(ge=1)
    cfg_dropout_probability: float = Field(ge=0.0, lt=1.0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    vocab_size: int = Field(ge=1)
    cnn: CNNConfig
    gru: GRUConfig
    transformer: TransformerConfig
    conditioning: ConditioningConfig
