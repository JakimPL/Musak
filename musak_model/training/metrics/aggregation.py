from __future__ import annotations

from dataclasses import dataclass
from math import exp

from musak_model.training.metrics.schema import BatchMetrics, EpochSplitMetrics


@dataclass
class MetricsAccumulator:
    loss_sum: float = 0.0
    token_count: int = 0
    exact_match_count: int = 0
    token_kind_match_count: int | None = None
    validity_penalty_loss_sum: float | None = None
    invalid_probability_mass_sum: float | None = None
    invalid_target_count: int | None = None
    validity_penalty_token_count: int | None = None
    cnn_gradient_norm_sum: float | None = None
    gru_gradient_norm_sum: float | None = None
    transformer_gradient_norm_sum: float | None = None

    def add(self, batch_metrics: BatchMetrics) -> None:
        self.loss_sum += batch_metrics.loss * batch_metrics.token_count
        self.token_count += batch_metrics.token_count
        self.exact_match_count += batch_metrics.exact_match_count
        if batch_metrics.token_kind_match_count is not None:
            if self.token_kind_match_count is None:
                self.token_kind_match_count = 0

            self.token_kind_match_count += batch_metrics.token_kind_match_count
        if batch_metrics.validity_penalty_token_count is not None:
            if self.validity_penalty_token_count is None:
                self.validity_penalty_token_count = 0
                self.validity_penalty_loss_sum = 0.0
                self.invalid_probability_mass_sum = 0.0
                self.invalid_target_count = 0

            self.validity_penalty_token_count += batch_metrics.validity_penalty_token_count
            self.validity_penalty_loss_sum = (self.validity_penalty_loss_sum or 0.0) + (
                batch_metrics.validity_penalty_loss or 0.0
            ) * batch_metrics.validity_penalty_token_count
            self.invalid_probability_mass_sum = (self.invalid_probability_mass_sum or 0.0) + (
                batch_metrics.invalid_probability_mass or 0.0
            ) * batch_metrics.validity_penalty_token_count
            self.invalid_target_count = (self.invalid_target_count or 0) + (batch_metrics.invalid_target_count or 0)
        self.cnn_gradient_norm_sum = _add_optional_weighted_metric(
            self.cnn_gradient_norm_sum,
            value=batch_metrics.cnn_gradient_norm,
            weight=batch_metrics.token_count,
        )
        self.gru_gradient_norm_sum = _add_optional_weighted_metric(
            self.gru_gradient_norm_sum,
            value=batch_metrics.gru_gradient_norm,
            weight=batch_metrics.token_count,
        )
        self.transformer_gradient_norm_sum = _add_optional_weighted_metric(
            self.transformer_gradient_norm_sum,
            value=batch_metrics.transformer_gradient_norm,
            weight=batch_metrics.token_count,
        )

    def to_epoch_split_metrics(self) -> EpochSplitMetrics:
        if self.token_count == 0:
            raise ValueError("cannot calculate metrics without tokens")

        loss = self.loss_sum / self.token_count
        return EpochSplitMetrics(
            loss=loss,
            perplexity=exp(loss),
            token_accuracy=self.exact_match_count / self.token_count,
            token_kind_accuracy=(
                None if self.token_kind_match_count is None else self.token_kind_match_count / self.token_count
            ),
            validity_penalty_loss=_optional_validity_average(
                self.validity_penalty_loss_sum,
                token_count=self.validity_penalty_token_count,
            ),
            invalid_probability_mass=_optional_validity_average(
                self.invalid_probability_mass_sum,
                token_count=self.validity_penalty_token_count,
            ),
            invalid_target_rate=(
                None if self.invalid_target_count is None else self.invalid_target_count / self.token_count
            ),
            cnn_gradient_norm=_weighted_optional_average(self.cnn_gradient_norm_sum, weight=self.token_count),
            gru_gradient_norm=_weighted_optional_average(self.gru_gradient_norm_sum, weight=self.token_count),
            transformer_gradient_norm=_weighted_optional_average(
                self.transformer_gradient_norm_sum,
                weight=self.token_count,
            ),
        )


def _add_optional_weighted_metric(current: float | None, *, value: float | None, weight: int) -> float | None:
    if value is None:
        return current

    return (current or 0.0) + value * weight


def _weighted_optional_average(value: float | None, *, weight: int) -> float | None:
    if value is None:
        return None

    return value / weight


def _optional_validity_average(value: float | None, *, token_count: int | None) -> float | None:
    if value is None or token_count is None or token_count == 0:
        return None

    return value / token_count
