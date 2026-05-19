from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor
from torch.nn import Module

from musak_model.training.checkpoint import CheckpointState

MigrationAction = Literal["copied", "initialized_missing", "expanded", "truncated"]


class CheckpointMigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MigratedTensor:
    key: str
    action: MigrationAction
    source_shape: tuple[int, ...] | None
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointMigrationReport:
    migrated_tensors: tuple[MigratedTensor, ...]
    ignored_source_keys: tuple[str, ...]
    optimizer_state_preserved: bool

    @property
    def changed_tensors(self) -> tuple[MigratedTensor, ...]:
        return tuple(tensor for tensor in self.migrated_tensors if tensor.action != "copied")


@dataclass(frozen=True)
class _SuccessfulTensorMigration:
    migrated_tensor: Tensor
    report: MigratedTensor


@dataclass(frozen=True)
class _FailedTensorMigration:
    error: str


type _TensorMigration = _SuccessfulTensorMigration | _FailedTensorMigration


def migrate_checkpoint_to_model(
    input_path: Path,
    output_path: Path,
    *,
    model: Module,
    device: torch.device,
    preserve_optimizer_state: bool = False,
    allow_truncation: bool = False,
) -> CheckpointMigrationReport:
    checkpoint = cast(CheckpointState, torch.load(input_path, map_location=device))
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    migrated_state: dict[str, Tensor] = {}
    migrated_tensors: list[MigratedTensor] = []
    errors: list[str] = []

    for key, target_tensor in target_state.items():
        migration = _migrate_tensor_to_target_shape(
            key,
            source_tensor=source_state.get(key),
            target_tensor=target_tensor,
            allow_truncation=allow_truncation,
        )
        match migration:
            case _SuccessfulTensorMigration():
                migrated_state[key] = migration.migrated_tensor
                migrated_tensors.append(migration.report)
            case _FailedTensorMigration():
                errors.append(migration.error)

    if errors:
        raise CheckpointMigrationError("\n".join(errors))

    ignored_source_keys = tuple(sorted(key for key in source_state if key not in target_state))
    migrated_checkpoint = dict(checkpoint)
    migrated_checkpoint["model_state_dict"] = migrated_state
    if not preserve_optimizer_state:
        migrated_checkpoint["optimizer_state_dict"] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated_checkpoint, output_path)
    return CheckpointMigrationReport(
        migrated_tensors=tuple(migrated_tensors),
        ignored_source_keys=ignored_source_keys,
        optimizer_state_preserved=preserve_optimizer_state,
    )


def _migrate_tensor_to_target_shape(
    key: str,
    *,
    source_tensor: Tensor | None,
    target_tensor: Tensor,
    allow_truncation: bool,
) -> _TensorMigration:
    if source_tensor is None:
        return _initialize_missing_target_weight(key, target_tensor)

    if source_tensor.shape == target_tensor.shape:
        return _copy_compatible_source_weight(key, source_tensor=source_tensor, target_tensor=target_tensor)

    return _adapt_shape_changed_source_weight(
        key,
        source_tensor=source_tensor,
        target_tensor=target_tensor,
        allow_truncation=allow_truncation,
    )


def _initialize_missing_target_weight(key: str, target_tensor: Tensor) -> _SuccessfulTensorMigration:
    return _SuccessfulTensorMigration(
        migrated_tensor=target_tensor.detach().clone(),
        report=MigratedTensor(
            key=key,
            action="initialized_missing",
            source_shape=None,
            target_shape=tuple(target_tensor.shape),
        ),
    )


def _copy_compatible_source_weight(
    key: str,
    *,
    source_tensor: Tensor,
    target_tensor: Tensor,
) -> _SuccessfulTensorMigration:
    return _SuccessfulTensorMigration(
        migrated_tensor=source_tensor.detach()
        .clone()
        .to(
            device=target_tensor.device,
            dtype=target_tensor.dtype,
        ),
        report=MigratedTensor(
            key=key,
            action="copied",
            source_shape=tuple(source_tensor.shape),
            target_shape=tuple(target_tensor.shape),
        ),
    )


def _adapt_shape_changed_source_weight(
    key: str,
    *,
    source_tensor: Tensor,
    target_tensor: Tensor,
    allow_truncation: bool,
) -> _TensorMigration:
    if source_tensor.ndim != target_tensor.ndim:
        return _FailedTensorMigration(
            error=(
                f"{key}: cannot migrate {tuple(source_tensor.shape)} to {tuple(target_tensor.shape)} "
                "because tensor ranks differ"
            )
        )

    source_shape = tuple(source_tensor.shape)
    target_shape = tuple(target_tensor.shape)
    truncates = _source_shape_requires_truncation(source_shape=source_shape, target_shape=target_shape)
    if truncates and not allow_truncation:
        return _FailedTensorMigration(
            error=f"{key}: refusing to truncate {source_shape} to {target_shape}; pass allow_truncation to permit it"
        )

    return _SuccessfulTensorMigration(
        migrated_tensor=_copy_overlap(source_tensor, target_tensor),
        report=MigratedTensor(
            key=key,
            action="truncated" if truncates else "expanded",
            source_shape=source_shape,
            target_shape=target_shape,
        ),
    )


def _source_shape_requires_truncation(
    *,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> bool:
    return any(source_size > target_size for source_size, target_size in zip(source_shape, target_shape))


def _copy_overlap(source_tensor: Tensor, target_tensor: Tensor) -> Tensor:
    migrated_tensor = target_tensor.detach().clone()
    overlap = tuple(
        slice(0, min(source_size, target_size))
        for source_size, target_size in zip(source_tensor.shape, target_tensor.shape)
    )
    migrated_tensor[overlap] = source_tensor.detach().to(
        device=target_tensor.device,
        dtype=target_tensor.dtype,
    )[overlap]
    return migrated_tensor
