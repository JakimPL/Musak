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
