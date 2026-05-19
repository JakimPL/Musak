from pathlib import Path
from typing import TypedDict, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


class CheckpointState(TypedDict):
    epoch: int
    best_validation_loss: float | None
    model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, object]


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_validation_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = CheckpointState(
        epoch=epoch,
        best_validation_loss=best_validation_loss,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
    )
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[int, float | None]:
    state = cast(CheckpointState, torch.load(path, map_location=device))
    model.load_state_dict(state["model_state_dict"])
    if _optimizer_state_matches_optimizer(state["optimizer_state_dict"], optimizer=optimizer):
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state["epoch"] + 1, state["best_validation_loss"]


def load_model_weights(
    path: Path,
    *,
    model: nn.Module,
    device: torch.device,
) -> None:
    state = cast(CheckpointState, torch.load(path, map_location=device))
    model.load_state_dict(state["model_state_dict"])


def _optimizer_state_matches_optimizer(
    optimizer_state_dict: dict[str, object],
    *,
    optimizer: Optimizer,
) -> bool:
    saved_groups = optimizer_state_dict.get("param_groups")
    current_groups = optimizer.state_dict().get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        return False

    if len(saved_groups) != len(current_groups):
        return False

    return all(
        _parameter_group_sizes_match(saved_group, current_group)
        for saved_group, current_group in zip(saved_groups, current_groups)
    )


def _parameter_group_sizes_match(saved_group: object, current_group: object) -> bool:
    if not isinstance(saved_group, dict) or not isinstance(current_group, dict):
        return False

    saved_parameters = saved_group.get("params")
    current_parameters = current_group.get("params")
    if not isinstance(saved_parameters, list) or not isinstance(current_parameters, list):
        return False

    return len(saved_parameters) == len(current_parameters)
