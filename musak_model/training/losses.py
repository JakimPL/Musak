from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from musak_model.model.output import FactorizedTokenLogits
from musak_model.tokens.factorized import ABSENT_ATTRIBUTE_ID
from musak_model.training.config import EventObjectiveConfig
from musak_model.training.dataset.factorized import TokenAttributeTargetTensors


@dataclass(frozen=True)
class FactorizedEventLoss:
    loss: Tensor
    kind_loss: Tensor
    duration_loss: Tensor
    degree_loss: Tensor
    accidental_loss: Tensor
    octave_offset_loss: Tensor
    hand_loss: Tensor
    kind_target_count: int
    duration_target_count: int
    degree_target_count: int
    accidental_target_count: int
    octave_offset_target_count: int
    hand_target_count: int


def factorized_event_loss(
    logits: FactorizedTokenLogits,
    *,
    targets: TokenAttributeTargetTensors,
    config: EventObjectiveConfig,
) -> FactorizedEventLoss:
    kind_loss, kind_target_count = _masked_cross_entropy(logits.kind, targets.kind_ids)
    duration_loss, duration_target_count = _masked_cross_entropy(logits.duration, targets.duration_ids)
    degree_loss, degree_target_count = _masked_cross_entropy(logits.degree, targets.degree_ids)
    accidental_loss, accidental_target_count = _masked_cross_entropy(logits.accidental, targets.accidental_ids)
    octave_offset_loss, octave_offset_target_count = _masked_cross_entropy(
        logits.octave_offset,
        targets.octave_offset_ids,
    )
    hand_loss, hand_target_count = _masked_cross_entropy(logits.hand, targets.hand_ids)
    loss = (
        config.kind_weight * kind_loss
        + config.duration_weight * duration_loss
        + config.degree_weight * degree_loss
        + config.accidental_weight * accidental_loss
        + config.octave_offset_weight * octave_offset_loss
        + config.hand_weight * hand_loss
    )
    return FactorizedEventLoss(
        loss=loss,
        kind_loss=kind_loss,
        duration_loss=duration_loss,
        degree_loss=degree_loss,
        accidental_loss=accidental_loss,
        octave_offset_loss=octave_offset_loss,
        hand_loss=hand_loss,
        kind_target_count=kind_target_count,
        duration_target_count=duration_target_count,
        degree_target_count=degree_target_count,
        accidental_target_count=accidental_target_count,
        octave_offset_target_count=octave_offset_target_count,
        hand_target_count=hand_target_count,
    )


def _masked_cross_entropy(logits: Tensor, targets: Tensor) -> tuple[Tensor, int]:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape[:-1])} does not match targets shape {tuple(targets.shape)}")

    flat_targets = targets.reshape(-1)
    active_mask = flat_targets != ABSENT_ATTRIBUTE_ID
    target_count = int(active_mask.sum().item())
    if target_count == 0:
        return logits.sum() * 0.0, 0

    flat_logits = logits.reshape(-1, logits.size(-1))
    active_targets = flat_targets[active_mask]
    if torch.any(active_targets < 0):
        raise ValueError("active factorized targets contain negative ids")

    if torch.any(active_targets >= logits.size(-1)):
        raise ValueError("active factorized targets contain ids outside the corresponding head range")

    return nn.functional.cross_entropy(flat_logits[active_mask], active_targets, reduction="mean"), target_count
