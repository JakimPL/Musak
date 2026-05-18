from __future__ import annotations

from math import sqrt

from torch import nn


def module_gradient_norm_metrics(model: nn.Module) -> dict[str, float | None]:
    return {
        "cnn_gradient_norm": _gradient_norm_for_prefixes(model, prefixes=("_to_local_hidden", "_local_encoder")),
        "gru_gradient_norm": _gradient_norm_for_prefixes(
            model,
            prefixes=(
                "_to_bar_hidden",
                "_bar_prefix_encoder",
                "_bar_encoder",
                "_bar_prefix_to_transformer_hidden",
                "_bar_to_transformer_hidden",
            ),
        ),
        "transformer_gradient_norm": _gradient_norm_for_prefixes(model, prefixes=("_decoder",)),
    }


def _gradient_norm_for_prefixes(model: nn.Module, *, prefixes: tuple[str, ...]) -> float | None:
    squared_norm_sum = 0.0
    has_gradient = False
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not name.startswith(prefixes):
            continue

        has_gradient = True
        gradient_norm = float(parameter.grad.detach().norm(2).item())
        squared_norm_sum += gradient_norm * gradient_norm

    if not has_gradient:
        return None

    return sqrt(squared_norm_sum)
