from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from musak_model.auxiliary.schema import (
    MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
    MusicalAuxiliaryLogits,
    MusicalAuxiliaryTargetTensors,
)
from musak_model.model.output import FactorizedTokenLogits
from musak_model.tokens.factorized import ABSENT_ATTRIBUTE_ID
from musak_model.training.config import EventObjectiveConfig, MusicalAuxiliaryObjectiveConfig
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


@dataclass(frozen=True)
class MusicalAuxiliaryLoss:
    loss: Tensor
    note_density_loss: Tensor
    rhythmic_diversity_loss: Tensor
    voice_independence_loss: Tensor
    uses_accidentals_loss: Tensor
    dotted_duration_loss: Tensor
    hand_span_loss: Tensor
    bar_note_density_loss: Tensor
    bar_rhythmic_diversity_loss: Tensor
    bar_voice_independence_loss: Tensor
    bar_uses_accidentals_loss: Tensor
    bar_dotted_duration_loss: Tensor
    bar_hand_span_loss: Tensor
    note_density_match_count: int
    note_density_target_count: int
    rhythmic_diversity_match_count: int
    rhythmic_diversity_target_count: int
    voice_independence_match_count: int
    voice_independence_target_count: int
    uses_accidentals_match_count: int
    uses_accidentals_target_count: int
    dotted_duration_match_count: int
    dotted_duration_target_count: int
    hand_span_match_count: int
    hand_span_target_count: int
    bar_note_density_match_count: int
    bar_note_density_target_count: int
    bar_rhythmic_diversity_match_count: int
    bar_rhythmic_diversity_target_count: int
    bar_voice_independence_match_count: int
    bar_voice_independence_target_count: int
    bar_uses_accidentals_match_count: int
    bar_uses_accidentals_target_count: int
    bar_dotted_duration_match_count: int
    bar_dotted_duration_target_count: int
    bar_hand_span_match_count: int
    bar_hand_span_target_count: int


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


def musical_auxiliary_loss(
    logits: MusicalAuxiliaryLogits,
    *,
    targets: MusicalAuxiliaryTargetTensors,
    config: MusicalAuxiliaryObjectiveConfig,
) -> MusicalAuxiliaryLoss:
    note_density_loss, note_density_match_count, note_density_target_count = _target_cross_entropy(
        logits.note_density,
        targets.note_density_ids,
    )
    rhythmic_diversity_loss, rhythmic_diversity_match_count, rhythmic_diversity_target_count = _target_cross_entropy(
        logits.rhythmic_diversity,
        targets.rhythmic_diversity_ids,
    )
    voice_independence_loss, voice_independence_match_count, voice_independence_target_count = _target_cross_entropy(
        logits.voice_independence,
        targets.voice_independence_ids,
    )
    uses_accidentals_loss, uses_accidentals_match_count, uses_accidentals_target_count = _target_cross_entropy(
        logits.uses_accidentals,
        targets.uses_accidentals_ids,
    )
    dotted_duration_loss, dotted_duration_match_count, dotted_duration_target_count = _target_cross_entropy(
        logits.dotted_duration,
        targets.dotted_duration_ids,
    )
    hand_span_loss, hand_span_match_count, hand_span_target_count = _target_cross_entropy(
        logits.hand_span,
        targets.hand_span_ids,
    )
    bar_note_density_loss, bar_note_density_match_count, bar_note_density_target_count = _target_cross_entropy(
        logits.bar.note_density,
        targets.bar_targets.note_density_ids,
    )
    bar_rhythmic_diversity_loss, bar_rhythmic_diversity_match_count, bar_rhythmic_diversity_target_count = (
        _target_cross_entropy(
            logits.bar.rhythmic_diversity,
            targets.bar_targets.rhythmic_diversity_ids,
        )
    )
    bar_voice_independence_loss, bar_voice_independence_match_count, bar_voice_independence_target_count = (
        _target_cross_entropy(
            logits.bar.voice_independence,
            targets.bar_targets.voice_independence_ids,
        )
    )
    bar_uses_accidentals_loss, bar_uses_accidentals_match_count, bar_uses_accidentals_target_count = (
        _target_cross_entropy(
            logits.bar.uses_accidentals,
            targets.bar_targets.uses_accidentals_ids,
        )
    )
    bar_dotted_duration_loss, bar_dotted_duration_match_count, bar_dotted_duration_target_count = _target_cross_entropy(
        logits.bar.dotted_duration,
        targets.bar_targets.dotted_duration_ids,
    )
    bar_hand_span_loss, bar_hand_span_match_count, bar_hand_span_target_count = _target_cross_entropy(
        logits.bar.hand_span,
        targets.bar_targets.hand_span_ids,
    )
    sequence_loss = (
        config.note_density_weight * note_density_loss
        + config.rhythmic_diversity_weight * rhythmic_diversity_loss
        + config.voice_independence_weight * voice_independence_loss
        + config.uses_accidentals_weight * uses_accidentals_loss
        + config.dotted_duration_weight * dotted_duration_loss
        + config.hand_span_weight * hand_span_loss
    )
    bar_loss = (
        config.note_density_weight * bar_note_density_loss
        + config.rhythmic_diversity_weight * bar_rhythmic_diversity_loss
        + config.voice_independence_weight * bar_voice_independence_loss
        + config.uses_accidentals_weight * bar_uses_accidentals_loss
        + config.dotted_duration_weight * bar_dotted_duration_loss
        + config.hand_span_weight * bar_hand_span_loss
    )
    return MusicalAuxiliaryLoss(
        loss=sequence_loss + config.bar_weight * bar_loss,
        note_density_loss=note_density_loss,
        rhythmic_diversity_loss=rhythmic_diversity_loss,
        voice_independence_loss=voice_independence_loss,
        uses_accidentals_loss=uses_accidentals_loss,
        dotted_duration_loss=dotted_duration_loss,
        hand_span_loss=hand_span_loss,
        bar_note_density_loss=bar_note_density_loss,
        bar_rhythmic_diversity_loss=bar_rhythmic_diversity_loss,
        bar_voice_independence_loss=bar_voice_independence_loss,
        bar_uses_accidentals_loss=bar_uses_accidentals_loss,
        bar_dotted_duration_loss=bar_dotted_duration_loss,
        bar_hand_span_loss=bar_hand_span_loss,
        note_density_match_count=note_density_match_count,
        note_density_target_count=note_density_target_count,
        rhythmic_diversity_match_count=rhythmic_diversity_match_count,
        rhythmic_diversity_target_count=rhythmic_diversity_target_count,
        voice_independence_match_count=voice_independence_match_count,
        voice_independence_target_count=voice_independence_target_count,
        uses_accidentals_match_count=uses_accidentals_match_count,
        uses_accidentals_target_count=uses_accidentals_target_count,
        dotted_duration_match_count=dotted_duration_match_count,
        dotted_duration_target_count=dotted_duration_target_count,
        hand_span_match_count=hand_span_match_count,
        hand_span_target_count=hand_span_target_count,
        bar_note_density_match_count=bar_note_density_match_count,
        bar_note_density_target_count=bar_note_density_target_count,
        bar_rhythmic_diversity_match_count=bar_rhythmic_diversity_match_count,
        bar_rhythmic_diversity_target_count=bar_rhythmic_diversity_target_count,
        bar_voice_independence_match_count=bar_voice_independence_match_count,
        bar_voice_independence_target_count=bar_voice_independence_target_count,
        bar_uses_accidentals_match_count=bar_uses_accidentals_match_count,
        bar_uses_accidentals_target_count=bar_uses_accidentals_target_count,
        bar_dotted_duration_match_count=bar_dotted_duration_match_count,
        bar_dotted_duration_target_count=bar_dotted_duration_target_count,
        bar_hand_span_match_count=bar_hand_span_match_count,
        bar_hand_span_target_count=bar_hand_span_target_count,
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


def _target_cross_entropy(logits: Tensor, targets: Tensor) -> tuple[Tensor, int, int]:
    if logits.shape[:-1] != targets.shape:
        logits_shape = tuple(logits.shape[:-1])
        targets_shape = tuple(targets.shape)
        raise ValueError(f"auxiliary logits shape {logits_shape} does not match targets shape {targets_shape}")

    flat_targets = targets.reshape(-1)
    active_mask = flat_targets != MUSICAL_AUXILIARY_TARGET_IGNORE_ID
    target_count = int(active_mask.sum().item())
    if target_count == 0:
        return logits.sum() * 0.0, 0, 0

    active_targets = flat_targets[active_mask]
    if torch.any(active_targets < 0):
        raise ValueError("active auxiliary targets contain negative ids")

    if torch.any(active_targets >= logits.size(-1)):
        raise ValueError("active auxiliary targets contain ids outside the corresponding head range")

    flat_logits = logits.reshape(-1, logits.size(-1))
    active_logits = flat_logits[active_mask]
    predictions = active_logits.argmax(dim=-1)
    match_count = int((predictions == active_targets).sum().item())
    return nn.functional.cross_entropy(active_logits, active_targets, reduction="mean"), match_count, target_count
