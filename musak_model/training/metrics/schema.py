from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EpochMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: int
    train_loss: float
    train_perplexity: float
    train_token_accuracy: float
    train_token_kind_accuracy: float | None = None
    train_validity_penalty_loss: float | None = None
    train_invalid_probability_mass: float | None = None
    train_invalid_target_rate: float | None = None
    train_cnn_gradient_norm: float | None = None
    train_gru_gradient_norm: float | None = None
    train_transformer_gradient_norm: float | None = None
    validation_loss: float | None
    validation_perplexity: float | None = None
    validation_token_accuracy: float | None = None
    validation_token_kind_accuracy: float | None = None
    validation_validity_penalty_loss: float | None = None
    validation_invalid_probability_mass: float | None = None
    validation_invalid_target_rate: float | None = None


class EpochSplitMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    loss: float
    perplexity: float
    token_accuracy: float
    token_kind_accuracy: float | None = None
    validity_penalty_loss: float | None = None
    invalid_probability_mass: float | None = None
    invalid_target_rate: float | None = None
    cnn_gradient_norm: float | None = None
    gru_gradient_norm: float | None = None
    transformer_gradient_norm: float | None = None


class BatchMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    loss: float
    token_count: int = Field(ge=1)
    exact_match_count: int = Field(ge=0)
    token_kind_match_count: int | None = None
    validity_penalty_loss: float | None = None
    invalid_probability_mass: float | None = None
    invalid_target_count: int | None = None
    validity_penalty_token_count: int | None = None
    cnn_gradient_norm: float | None = None
    gru_gradient_norm: float | None = None
    transformer_gradient_norm: float | None = None
