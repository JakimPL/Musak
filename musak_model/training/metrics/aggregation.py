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
